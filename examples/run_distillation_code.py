# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""On-policy distillation for code tasks (e.g. ICPC).

Differences from run_distillation_math.py:
  - Loads prompts from a local JSONL via data.train_data_path.
  - The JSONL rows are expected to have the prompt at data.prompt_key
    (default: "responses_create_params.input" for Nemo-Gym ICPC format,
     or a flat top-level key such as "prompt").
  - No environment-based validation is performed; set val_period: 0 in config.
"""

import argparse
import os
from collections import defaultdict
from typing import Any, Optional

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.distillation import MasterConfig, distillation_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

TokenizerType = PreTrainedTokenizerBase

DEFAULT_DISTILLATION_TEACHER_BIAS_TEMPLATE = (
    "Problem:\n"
    "{problem}\n\n"
    "Here is a reference solution:\n"
    "{reference}\n\n"
    "After understanding the reference solution, please try to solve this problem\n"
    "using your own approach below:\n"
    "Answer:\n"
)


def _build_prompt_message(prompt: str, tokenizer: TokenizerType) -> dict[str, Any]:
    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    formatted: str = tokenizer.apply_chat_template(  # type: ignore
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    user_message["content"] = formatted
    return user_message


# ===============================================================================
#                          Code Data Processor
# ===============================================================================


def code_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process an ICPC/code datum into a DatumSpec.

    Expects datum_dict to contain:
      - "prompt": the fully-formatted problem string (already includes system
                  instructions and problem statement)
      - "task_name": task identifier (e.g. "icpc")
    """
    prompt = datum_dict["prompt"]

    message_log: LLMMessageLogType = [_build_prompt_message(prompt, tokenizer)]

    teacher_message_log: Optional[LLMMessageLogType] = None
    teacher_reference = datum_dict.get("teacher_reference")
    if isinstance(teacher_reference, str) and teacher_reference.strip():
        teacher_bias_template = datum_dict.get(
            "teacher_bias_template",
            DEFAULT_DISTILLATION_TEACHER_BIAS_TEMPLATE,
        )
        try:
            teacher_prompt = str(teacher_bias_template).format(
                problem=prompt,
                reference=teacher_reference.strip(),
            )
        except Exception:
            teacher_prompt = DEFAULT_DISTILLATION_TEACHER_BIAS_TEMPLATE.format(
                problem=prompt,
                reference=teacher_reference.strip(),
            )
        teacher_message_log = [_build_prompt_message(teacher_prompt, tokenizer)]

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if length > max_seq_length:
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    datum = DatumSpec(
        message_log=message_log,
        length=length,
        extra_env_info={
            "ground_truth": datum_dict.get("teacher_reference", ""),
        },
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name=datum_dict["task_name"],
    )
    if teacher_message_log is not None:
        datum["teacher_message_log"] = teacher_message_log
        datum["teacher_replace_message_count"] = len(message_log)
    return datum


def _extract_nested(
    row: dict[str, Any],
    dotted_key: str,
    default: Optional[Any] = None,
) -> Any:
    """Resolve a dotted key such as 'responses_create_params.input'."""
    parts = dotted_key.split(".")
    value: Any = row
    try:
        for part in parts:
            value = value[part]
    except (KeyError, TypeError):
        return default
    return value


# ===============================================================================
#                             Data Setup
# ===============================================================================


def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up code distillation data...")

    train_path = data_config["train_data_path"]
    prompt_key = data_config.get("prompt_key", "responses_create_params.input")
    reference_key = data_config.get("reference_key", "ground_truth_solution")
    teacher_bias_template = data_config.get(
        "teacher_bias_template",
        DEFAULT_DISTILLATION_TEACHER_BIAS_TEMPLATE,
    )
    task_name = "icpc"

    # Load the JSONL and extract the prompt field (supports dotted nested keys)
    print(f"  Loading data from: {train_path}")
    raw_ds = load_dataset("json", data_files=train_path)["train"]

    def _process_row(row: dict[str, Any]) -> dict[str, Any]:
        processed = {
            "prompt": _extract_nested(row, prompt_key),
            "task_name": task_name,
        }
        teacher_reference = _extract_nested(row, reference_key)
        if isinstance(teacher_reference, str) and teacher_reference.strip():
            processed["teacher_reference"] = teacher_reference.strip()
            processed["teacher_bias_template"] = teacher_bias_template
        return processed

    ds = raw_ds.map(_process_row)

    if data_config.get("shuffle", True):
        ds = ds.shuffle(seed=seed)

    code_task_spec = TaskDataSpec(
        task_name=task_name,
        prompt_file=None,
        system_prompt_file=None,
    )

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (code_task_spec, code_data_processor))
    )
    task_data_processors[task_name] = (code_task_spec, code_data_processor)

    dataset = AllTaskProcessedDataset(
        ds,
        tokenizer,
        code_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    # Validation is disabled for code tasks (requires sandbox execution).
    # Set distillation.val_period: 0 in config to skip validation entirely.
    val_dataset = None

    # Use MathEnvironment as a placeholder; it is only invoked if val_period > 0.
    math_env_cfg = env_configs.get("math", {"num_workers": 1})
    dummy_env = MathEnvironment.options(  # type: ignore
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(math_env_cfg)

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: dummy_env)
    task_to_env[task_name] = dummy_env

    return dataset, val_dataset, task_to_env, task_to_env


# ===============================================================================
#                                   Main
# ===============================================================================


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run code distillation training with configuration"
    )
    parser.add_argument("--config", type=str, default=None)
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "distillation_icpc_qwen3_0_6b_from_1_7b.yaml",
        )

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    if config["policy"]["generation"] is not None:
        config["policy"]["generation"] = configure_generation_config(
            config["policy"]["generation"], tokenizer
        )
    else:
        print("  ⚠️  No generation config found, this may cause issues")

    dataset, val_dataset, task_to_env, val_task_to_env = setup_data(
        tokenizer, config["data"], config["env"], 42
    )

    (
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    distillation_train(
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    )


if __name__ == "__main__":
    main()
