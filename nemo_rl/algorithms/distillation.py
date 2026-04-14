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
# See the License for the specific language governing permissions and limitations.
# limitations under the License.
import json
import math
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import _should_use_async_rollouts, refit_policy_generation
from nemo_rl.algorithms.loss_functions import (
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_nemo_gym_rollout,
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class DistillationConfig(TypedDict):
    # Training configuration
    num_prompts_per_step: int
    num_generations_per_prompt: int
    rollout_greedy: NotRequired[bool]
    filter_truncated_rollouts: NotRequired[bool]
    truncated_rollout_tolerance: NotRequired[float]
    filter_repetitive_rollouts: NotRequired[bool]
    repetition_min_response_tokens: NotRequired[int]
    repetition_max_token_run: NotRequired[int]
    repetition_max_token_fraction: NotRequired[float]
    require_code_block: NotRequired[bool]
    max_rollout_turns: int  # for multi-turn rollouts. Math Environments just have 1 turn (answering the question)
    max_num_steps: int  # maximum number of steps to train for
    max_num_epochs: int  # maximum number of epochs to train for
    val_batch_size: int
    val_period: int
    val_at_start: bool
    max_val_samples: int
    topk_logits_k: int
    seed: int
    teacher_update_mode: NotRequired[str]
    teacher_update_period: NotRequired[int]


class DistillationSaveState(TypedDict):
    total_steps: int  # Track total number of steps across all epochs
    current_epoch: int  # Track current epoch
    current_step: int  # Track step within current epoch
    val_reward: NotRequired[
        float
    ]  # Can be any metric. Setted to 'accuracy' by default in validation.
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training


def _default_distillation_save_state() -> DistillationSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "val_reward": -99999999.0,  # Aligned with GRPO
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


class MasterConfig(TypedDict):
    """Main configuration structure."""

    policy: PolicyConfig  # Student model configuration
    teacher: PolicyConfig  # Teacher model configuration
    loss_fn: DistillationLossConfig  # Loss function configuration
    env: dict[str, Any]  # Environment configuration
    data: DataConfig  # Data configuration
    distillation: DistillationConfig  # Distillation configuration
    logger: LoggerConfig  # Logger configuration
    cluster: ClusterConfig  # Cluster configuration
    checkpointing: CheckpointingConfig  # Checkpointing configuration


# ===============================================================================
# Setup & Initialization
# ===============================================================================
def check_vocab_equality(
    tokenizer: TokenizerType, student_model_name: str, teacher_model_name: str
) -> None:
    """Check if the vocab of the tokenizer (student) and the teacher tokenizer are equal."""
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    skip_hint = "Set NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true to skip this check."

    # 1) Exact token->id mapping equality
    vocab_a = tokenizer.get_vocab()
    vocab_b = teacher_tokenizer.get_vocab()
    assert vocab_a == vocab_b, (
        f"Token->ID mapping differs between student and teacher. {skip_hint}"
    )

    # 2) Size consistency (sanity checks)
    assert len(tokenizer) == len(teacher_tokenizer), (
        f"Effective vocab sizes differ between student and teacher. {skip_hint}"
    )

    # 3) Chech model.config.vocab_size to guarantee the last dimension of the logits is the same
    student_config = AutoConfig.from_pretrained(student_model_name)
    teacher_config = AutoConfig.from_pretrained(teacher_model_name)
    assert student_config.vocab_size == teacher_config.vocab_size, (
        f"Model config vocab sizes differ between student and teacher. {skip_hint}"
    )


def _get_teacher_update_mode(distillation_config: DistillationConfig) -> str:
    mode = distillation_config.get("teacher_update_mode", "fixed")
    if mode not in {"fixed", "periodic_sync"}:
        raise ValueError(
            "distillation.teacher_update_mode must be one of "
            "{'fixed', 'periodic_sync'}."
        )
    return mode


def _get_teacher_update_period(distillation_config: DistillationConfig) -> int:
    period = int(distillation_config.get("teacher_update_period", 1))
    if period < 1:
        raise ValueError("distillation.teacher_update_period must be >= 1.")
    return period


def _should_use_periodic_reference_teacher(
    policy_config: PolicyConfig,
    teacher_config: PolicyConfig,
    distillation_config: DistillationConfig,
) -> bool:
    mode = _get_teacher_update_mode(distillation_config)
    _get_teacher_update_period(distillation_config)
    if mode != "periodic_sync":
        return False
    if teacher_config["model_name"] != policy_config["model_name"]:
        raise ValueError(
            "distillation.teacher_update_mode=periodic_sync is only supported "
            "when teacher.model_name matches policy.model_name."
        )
    return True


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,  # student_policy
    Optional[ColocatablePolicyInterface],  # teacher_policy
    Optional[GenerationInterface],  # student_generation
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    DistillationLossFn,
    Logger,
    CheckpointManager,
    DistillationSaveState,
    MasterConfig,
]:
    """Main entry point for distillation algorithm.

    Returns:
        tuple of student_policy, teacher_policy, student_generation,
        train_dataloader, val_dataloader,
        loss_fn, logger, checkpointer, distillation_save_state, master_config
    """
    # Extract configuration
    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    distillation_config = master_config["distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    teacher_update_period = _get_teacher_update_period(distillation_config)
    use_periodic_reference_teacher = _should_use_periodic_reference_teacher(
        policy_config,
        teacher_config,
        distillation_config,
    )

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for distillation"
    )

    # Disallow SP + packing for dtensor path
    for cfg, who in ((policy_config, "student"), (teacher_config, "teacher")):
        # DTensor sequence parallel is supported; ensure CP and SP are not enabled together
        # This incompatibility is enforced in DTensor workers during initialization.
        # Additionally, SP may not be compatible with sequence packing for some models.
        # Refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details.
        # Therefore, we disable SP + packing for distillation.
        dtensor_enabled = cfg["dtensor_cfg"]["enabled"]
        sequence_packing_enabled = (
            "sequence_packing" in cfg and cfg["sequence_packing"]["enabled"]
        )
        sequence_parallel_enabled = (
            "sequence_parallel" in cfg["dtensor_cfg"]
            and cfg["dtensor_cfg"]["sequence_parallel"]
        )

        if dtensor_enabled and sequence_packing_enabled and sequence_parallel_enabled:
            raise AssertionError(
                f"Distillation does not support DTensor sequence parallel + sequence packing ({who} policy). "
                "Please refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details."
            )

    # Set random seed
    set_seed(distillation_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    distillation_save_state: Optional[DistillationSaveState] = cast(
        Optional[DistillationSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if distillation_save_state is None:
        distillation_save_state = _default_distillation_save_state()

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )

    if last_checkpoint_path:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(
        f"  ✓ Training dataloader loaded with {len(train_dataset)} samples", flush=True
    )

    # Load validation dataset if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    # If validation is enabled, load the validation dataloader
    if distillation_config["val_period"] > 0 or distillation_config["val_at_start"]:
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="distillation_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
            * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 3,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(
            f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes",
            flush=True,
        )
    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if cluster_config["num_nodes"] == 1:
            assert (
                inference_gpus_per_node is not None and inference_gpus_per_node > 0
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set to a value > 0 "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is not None
                and inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set and equal to cluster.gpus_per_node "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got inference_gpus_per_node={inference_gpus_per_node}, cluster.gpus_per_node={cluster_config['gpus_per_node']}."
            )
            train_nodes -= inference_nodes

        # create clusters
        train_cluster = RayVirtualCluster(
            name="distillation_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="distillation_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        print(
            f"  ✓ Separate clusters created: train={train_nodes}x{train_gpus_per_node}GPUs, inference={inference_nodes}x{inference_gpus_per_node}GPUs",
            flush=True,
        )

    # ==========================
    #      Teacher Policy
    # ==========================
    teacher_policy: Optional[ColocatablePolicyInterface] = None
    # Checkpoint paths
    weights_path = None
    optimizer_path = None

    if not bool(os.getenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", False)):
        check_vocab_equality(
            tokenizer, policy_config["model_name"], teacher_config["model_name"]
        )

    if "megatron_cfg" in teacher_config and teacher_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        teacher_config["megatron_cfg"]["train_iters"] = total_train_iters

    if use_periodic_reference_teacher:
        print(
            "\n▶ Using the student reference snapshot as the teacher "
            f"(sync every {teacher_update_period} step(s))...",
            flush=True,
        )
        if last_checkpoint_path:
            print(
                "  ⚠️ Resuming periodic self-distillation does not restore the last "
                "synced teacher snapshot; it is reinitialized during worker startup "
                "until the next sync.",
                flush=True,
            )
    else:
        print("\n▶ Setting up teacher policy...", flush=True)
        teacher_policy = Policy(
            name_prefix="teacher",
            cluster=train_cluster,
            config=teacher_config,
            tokenizer=tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=False,
            init_reference_model=False,
        )
        teacher_policy.offload_after_refit()

    # ==========================
    #    Student Generation Interface
    # ==========================
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        student_generation = None
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if "vllm_cfg" in generation_config:
            ## make vllm hf overrides match the training policy
            generation_config["vllm_cfg"]["hf_overrides"] = policy_config.get(
                "hf_config_overrides", {}
            )
        student_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        student_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}",
            flush=True,
        )

    # ==========================
    #      Student Policy
    # ==========================
    print("\n▶ Setting up student policy...", flush=True)

    # Checkpoint paths
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    student_policy = Policy(
        name_prefix="student",
        cluster=train_cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=use_periodic_reference_teacher,
    )

    if student_generation is not None:
        state_dict_info = student_policy.prepare_refit_info()
        student_generation.prepare_refit_info(state_dict_info)

    # if it is not colocated inference, initialize collective communication for update weights
    if not colocated_inference:
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
        train_world_size = train_cluster.world_size()
        # inference cluster + head node of the train cluster
        world_size = train_world_size + inference_nodes * inference_gpus_per_node
        # init collective
        futures_train = student_policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = student_generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )  # type: ignore
        # wait for all futures to complete
        ray.get(futures_train + futures_inference)

    loss_fn = DistillationLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n", flush=True)

    return (
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_save_state,
        master_config,
    )


# ===============================================================================
# Training & Validation
# ===============================================================================

LOG_SAMPLE_PERIOD = 1  # log debug samples every N steps (set to 0 to disable)
LOG_NUM_SAMPLES = 2    # number of decoded samples to print per debug step
ROLLOUT_VERIFICATION_APPEND_FAILED = False
CODE_LIKE_MARKERS = (
    "```",
    "```cpp",
    "```c++",
    "#include",
    "int main(",
    "using namespace std",
    "std::",
    "vector<",
    "cin >>",
    "cout <<",
)


def _log_debug_samples(
    step: int,
    tokenizer: Any,
    train_data: Any,
    repeated_batch: Any,
    invalid_samples: list[int],
) -> None:
    """Print per-step sample diagnostics to stdout.

    Outputs every LOG_SAMPLE_PERIOD steps. Shows:
      - sample_mask and token_mask statistics
      - fraction of student tokens found in teacher's top-k
      - teacher top-1 logit statistics at loss positions
      - per-sample generation length
      - decoded prompt + response for LOG_NUM_SAMPLES samples
    """
    if LOG_SAMPLE_PERIOD <= 0 or step % LOG_SAMPLE_PERIOD != 0:
        return

    print(f"\n{'#' * 70}", flush=True)
    print(f"[SAMPLE DEBUG] Step {step + 1}", flush=True)
    print(f"{'#' * 70}", flush=True)

    sample_mask = train_data.get("sample_mask")
    token_mask = train_data.get("token_mask")
    teacher_topk_indices = train_data.get("teacher_topk_indices")
    teacher_topk_logits = train_data.get("teacher_topk_logits")
    input_ids = train_data.get("input_ids")

    # sample_mask stats
    if sample_mask is not None:
        valid = int((sample_mask > 0).sum().item())
        total = int(sample_mask.numel())
        print(
            f"[SAMPLE DEBUG] sample_mask: {valid}/{total} valid "
            f"({100.0 * valid / max(1, total):.1f}%)",
            flush=True,
        )
    if invalid_samples:
        print(
            f"[SAMPLE DEBUG] invalid_samples (teacher/student token-count mismatch): "
            f"{len(invalid_samples)}  indices={invalid_samples[:10]}",
            flush=True,
        )

    # token_mask stats
    if token_mask is not None:
        valid_toks = int(token_mask.sum().item())
        total_toks = int(token_mask.numel())
        print(
            f"[SAMPLE DEBUG] token_mask: {valid_toks}/{total_toks} loss tokens "
            f"({100.0 * valid_toks / max(1, total_toks):.3f}%)",
            flush=True,
        )

    # teacher-student token overlap
    if (
        teacher_topk_indices is not None
        and input_ids is not None
        and token_mask is not None
        and token_mask.shape[1] > 1
    ):
        # token_mask[b,t]=1 → position t is a loss token; teacher logit at t predicts token t+1
        loss_mask_shift = token_mask[:, 1:].bool()   # [B, S-1]
        student_next = input_ids[:, 1:]              # [B, S-1]
        teacher_topk_shift = teacher_topk_indices[:, :-1, :]  # [B, S-1, k]

        student_in_topk = (teacher_topk_shift == student_next.unsqueeze(-1)).any(dim=-1)
        n_loss = int(loss_mask_shift.sum().item())
        if n_loss > 0:
            overlap = float((student_in_topk & loss_mask_shift).sum().item()) / n_loss
            topk = int(teacher_topk_indices.shape[-1])
            print(
                f"[SAMPLE DEBUG] student token in teacher top-{topk}: "
                f"{overlap * 100:.1f}% of {n_loss} loss positions",
                flush=True,
            )
        if teacher_topk_logits is not None and n_loss > 0:
            top1_logit = teacher_topk_logits[:, :, 0]
            masked_logits = top1_logit[token_mask.bool()]
            print(
                f"[SAMPLE DEBUG] teacher top-1 logit at loss positions: "
                f"mean={masked_logits.mean().item():.3f}  "
                f"min={masked_logits.min().item():.3f}  "
                f"max={masked_logits.max().item():.3f}",
                flush=True,
            )

    # per-sample generation lengths
    message_logs_list = repeated_batch.get("message_log", [])
    gen_lengths = []
    for msg_log in message_logs_list:
        asst_len = sum(
            int(m["token_ids"].numel())
            for m in msg_log
            if isinstance(m.get("token_ids"), torch.Tensor) and m.get("role") == "assistant"
        )
        gen_lengths.append(asst_len)
    if gen_lengths:
        print(
            f"[SAMPLE DEBUG] generation length: "
            f"mean={sum(gen_lengths)/len(gen_lengths):.1f}  "
            f"min={min(gen_lengths)}  max={max(gen_lengths)}",
            flush=True,
        )

    # decoded samples
    for i in range(min(LOG_NUM_SAMPLES, len(message_logs_list))):
        print(f"\n[SAMPLE DEBUG] ── Sample {i + 1} ──", flush=True)
        for msg in message_logs_list[i]:
            role = msg.get("role", "unknown")
            tids = msg.get("token_ids")
            if not isinstance(tids, torch.Tensor):
                continue
            decoded = tokenizer.decode(tids.tolist(), skip_special_tokens=False)
            n_tok = int(tids.numel())
            if role == "user":
                display = decoded[:400] + ("…" if len(decoded) > 400 else "")
                print(
                    f"[SAMPLE DEBUG] PROMPT ({n_tok} tok, first 400 chars):\n{display}",
                    flush=True,
                )
            elif role == "assistant":
                if len(decoded) > 800:
                    display = (
                        decoded[:500]
                        + f"\n…[{len(decoded) - 700} chars omitted]…\n"
                        + decoded[-200:]
                    )
                else:
                    display = decoded
                print(
                    f"[SAMPLE DEBUG] RESPONSE ({n_tok} tok):\n{display}",
                    flush=True,
                )
        sm_val = (
            sample_mask[i].item()
            if sample_mask is not None and i < len(sample_mask)
            else "n/a"
        )
        print(f"[SAMPLE DEBUG] sample_mask[{i}] = {sm_val}", flush=True)

    print(f"{'#' * 70}\n", flush=True)


def _apply_assistant_loss_mask(message_logs: list[Any]) -> None:
    for message_log in message_logs:
        for message in message_log:
            token_ids = message.get("token_ids")
            if not isinstance(token_ids, torch.Tensor):
                continue
            if message.get("role") == "assistant":
                message["token_loss_mask"] = torch.ones_like(token_ids)
            else:
                message["token_loss_mask"] = torch.zeros_like(token_ids)


def _get_last_assistant_message(
    message_log: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    for message in reversed(message_log):
        if message.get("role") == "assistant":
            return message
    return None


def _message_to_text(
    message: dict[str, Any], tokenizer: TokenizerType
) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    token_ids = message.get("token_ids")
    if isinstance(token_ids, torch.Tensor):
        return tokenizer.decode(token_ids.tolist(), skip_special_tokens=False)
    return ""


def _message_log_to_role_texts(
    message_log: list[dict[str, Any]],
    tokenizer: TokenizerType,
    role: str,
) -> list[str]:
    texts: list[str] = []
    for message in message_log:
        if message.get("role") != role:
            continue
        text = _message_to_text(message, tokenizer)
        if text:
            texts.append(text)
    return texts


def _get_rollout_verification_log_path() -> Path:
    override_path = os.getenv("NRL_DISTILL_VERIFY_JSONL")
    if override_path:
        return Path(override_path).expanduser()

    job_id = os.getenv("SLURM_JOB_ID", "local")
    return Path(os.getcwd()) / f"{job_id}-logs" / "ccc_verify.jsonl"


def _code_like_markers_present(text: str) -> list[str]:
    return [marker for marker in CODE_LIKE_MARKERS if marker in text]


def _token_repetition_stats(token_ids: torch.Tensor) -> tuple[int, float]:
    if token_ids.numel() <= 0:
        return 0, 0.0

    max_run = 1
    current_run = 1
    prev_token = int(token_ids[0].item())
    for token in token_ids[1:]:
        token_value = int(token.item())
        if token_value == prev_token:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
            prev_token = token_value

    _, counts = torch.unique(token_ids, return_counts=True)
    dominant_fraction = float(counts.max().item()) / float(token_ids.numel())
    return max_run, dominant_fraction


def _filter_pathological_rollouts(
    repeated_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    distillation_config: DistillationConfig,
    generation_max_new_tokens: int,
) -> dict[str, float]:
    sample_mask = repeated_batch.get("loss_multiplier")
    message_logs = repeated_batch.get("message_log")
    if not isinstance(sample_mask, torch.Tensor) or not isinstance(message_logs, list):
        return {}

    filter_truncated = bool(distillation_config.get("filter_truncated_rollouts", False))
    truncated_tolerance = float(
        distillation_config.get("truncated_rollout_tolerance", 0.98)
    )
    filter_repetitive = bool(
        distillation_config.get("filter_repetitive_rollouts", False)
    )
    repetition_min_tokens = int(
        distillation_config.get("repetition_min_response_tokens", 128)
    )
    repetition_max_run = int(distillation_config.get("repetition_max_token_run", 96))
    repetition_max_fraction = float(
        distillation_config.get("repetition_max_token_fraction", 0.25)
    )
    require_code_block = bool(distillation_config.get("require_code_block", False))

    filtered_truncated = 0
    filtered_repetitive = 0
    filtered_missing_code = 0
    filtered_missing_assistant = 0

    for sample_idx, message_log in enumerate(message_logs):
        if sample_idx >= len(sample_mask) or float(sample_mask[sample_idx].item()) <= 0:
            continue

        assistant_message = _get_last_assistant_message(message_log)
        if assistant_message is None:
            sample_mask[sample_idx] = 0
            filtered_missing_assistant += 1
            continue

        token_ids = assistant_message.get("token_ids")
        response_len = int(token_ids.numel()) if isinstance(token_ids, torch.Tensor) else 0
        if (
            filter_truncated
            and generation_max_new_tokens > 0
            and response_len >= max(1, int(truncated_tolerance * generation_max_new_tokens))
        ):
            sample_mask[sample_idx] = 0
            filtered_truncated += 1
            continue

        if filter_repetitive and isinstance(token_ids, torch.Tensor):
            max_run, dominant_fraction = _token_repetition_stats(token_ids)
            if (
                response_len >= repetition_min_tokens
                and (
                    max_run >= repetition_max_run
                    or dominant_fraction >= repetition_max_fraction
                )
            ):
                sample_mask[sample_idx] = 0
                filtered_repetitive += 1
                continue

        if require_code_block:
            response_text = _message_to_text(assistant_message, tokenizer)
            if "```" not in response_text:
                sample_mask[sample_idx] = 0
                filtered_missing_code += 1

    filtered_total = (
        filtered_truncated
        + filtered_repetitive
        + filtered_missing_code
        + filtered_missing_assistant
    )
    if filtered_total > 0:
        total_samples = int(sample_mask.numel())
        valid_after_filter = int((sample_mask > 0).sum().item())
        print(
            "⚠️ Filtered rollout targets before distillation: "
            f"{filtered_total}/{total_samples} "
            f"(truncated={filtered_truncated}, repetitive={filtered_repetitive}, "
            f"missing_code={filtered_missing_code}, missing_assistant={filtered_missing_assistant}).",
            flush=True,
        )
        return {
            "filtered_rollouts": float(filtered_total),
            "filtered_rollouts_truncated": float(filtered_truncated),
            "filtered_rollouts_repetitive": float(filtered_repetitive),
            "filtered_rollouts_missing_code": float(filtered_missing_code),
            "filtered_rollouts_missing_assistant": float(filtered_missing_assistant),
            "valid_rollouts_after_filter": float(valid_after_filter),
        }

    return {
        "filtered_rollouts": 0.0,
        "filtered_rollouts_truncated": 0.0,
        "filtered_rollouts_repetitive": 0.0,
        "filtered_rollouts_missing_code": 0.0,
        "filtered_rollouts_missing_assistant": 0.0,
        "valid_rollouts_after_filter": float((sample_mask > 0).sum().item()),
    }


def _build_rollout_verification_rows(
    *,
    step: int,
    tokenizer: TokenizerType,
    repeated_batch: BatchedDataDict[DatumSpec],
    distillation_config: DistillationConfig,
    generation_max_new_tokens: int,
    sample_mask_before_rollout_filters: Optional[torch.Tensor],
    sample_mask_after_rollout_filters: Optional[torch.Tensor],
    final_sample_mask: Optional[torch.Tensor],
    invalid_samples: list[int],
) -> list[dict[str, Any]]:
    message_logs = repeated_batch.get("message_log")
    if not isinstance(message_logs, list):
        return []

    filter_truncated = bool(distillation_config.get("filter_truncated_rollouts", False))
    truncated_tolerance = float(
        distillation_config.get("truncated_rollout_tolerance", 0.98)
    )
    filter_repetitive = bool(
        distillation_config.get("filter_repetitive_rollouts", False)
    )
    repetition_min_tokens = int(
        distillation_config.get("repetition_min_response_tokens", 128)
    )
    repetition_max_run = int(distillation_config.get("repetition_max_token_run", 96))
    repetition_max_fraction = float(
        distillation_config.get("repetition_max_token_fraction", 0.25)
    )
    require_code_block = bool(distillation_config.get("require_code_block", False))
    invalid_sample_set = set(invalid_samples)
    teacher_prompt_message_logs = repeated_batch.get("teacher_message_log")

    rows: list[dict[str, Any]] = []
    for sample_idx, message_log in enumerate(message_logs):
        prompt_texts = _message_log_to_role_texts(message_log, tokenizer, "user")
        assistant_texts = _message_log_to_role_texts(message_log, tokenizer, "assistant")
        teacher_prompt_texts: list[str] = []
        if (
            isinstance(teacher_prompt_message_logs, list)
            and sample_idx < len(teacher_prompt_message_logs)
            and isinstance(teacher_prompt_message_logs[sample_idx], list)
        ):
            teacher_prompt_texts = _message_log_to_role_texts(
                teacher_prompt_message_logs[sample_idx], tokenizer, "user"
            )

        assistant_message = _get_last_assistant_message(message_log)
        response_text = (
            _message_to_text(assistant_message, tokenizer)
            if assistant_message is not None
            else ""
        )
        token_ids = assistant_message.get("token_ids") if assistant_message else None
        response_len = int(token_ids.numel()) if isinstance(token_ids, torch.Tensor) else 0

        truncated = bool(
            assistant_message is not None
            and filter_truncated
            and generation_max_new_tokens > 0
            and response_len
            >= max(1, int(truncated_tolerance * generation_max_new_tokens))
        )
        repetitive = False
        max_run = 0
        dominant_fraction = 0.0
        if assistant_message is not None and filter_repetitive and isinstance(
            token_ids, torch.Tensor
        ):
            max_run, dominant_fraction = _token_repetition_stats(token_ids)
            repetitive = bool(
                response_len >= repetition_min_tokens
                and (
                    max_run >= repetition_max_run
                    or dominant_fraction >= repetition_max_fraction
                )
            )

        code_like_markers = _code_like_markers_present(response_text)
        has_code_block = "```" in response_text
        missing_code = bool(
            assistant_message is not None and require_code_block and not has_code_block
        )

        filter_reasons: list[str] = []
        if assistant_message is None:
            filter_reasons.append("missing_assistant")
        else:
            if truncated:
                filter_reasons.append("truncated")
            if repetitive:
                filter_reasons.append("repetitive")
            if missing_code:
                filter_reasons.append("missing_code")
        if sample_idx in invalid_sample_set:
            filter_reasons.append("teacher_student_token_count_mismatch")

        rows.append(
            {
                "step": step + 1,
                "sample_idx": sample_idx,
                "loss_multiplier_before_rollout_filters": (
                    float(sample_mask_before_rollout_filters[sample_idx].item())
                    if sample_mask_before_rollout_filters is not None
                    and sample_idx < len(sample_mask_before_rollout_filters)
                    else None
                ),
                "loss_multiplier_after_rollout_filters": (
                    float(sample_mask_after_rollout_filters[sample_idx].item())
                    if sample_mask_after_rollout_filters is not None
                    and sample_idx < len(sample_mask_after_rollout_filters)
                    else None
                ),
                "loss_multiplier_final": (
                    float(final_sample_mask[sample_idx].item())
                    if final_sample_mask is not None and sample_idx < len(final_sample_mask)
                    else None
                ),
                "filter_reasons": filter_reasons,
                "had_assistant_message": assistant_message is not None,
                "invalid_teacher_alignment": sample_idx in invalid_sample_set,
                "response_token_count": response_len,
                "generation_max_new_tokens": generation_max_new_tokens,
                "requires_code_block": require_code_block,
                "has_code_block": has_code_block,
                "code_fence_count": response_text.count("```"),
                "code_like_markers": code_like_markers,
                "looks_like_code": len(code_like_markers) > 0,
                "max_repeated_token_run": max_run,
                "dominant_token_fraction": dominant_fraction,
                "prompt_text": "\n\n".join(prompt_texts),
                "teacher_prompt_text": "\n\n".join(teacher_prompt_texts),
                "response_text": response_text,
                "assistant_texts": assistant_texts,
            }
        )

    return rows


def _append_rollout_verification_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    global ROLLOUT_VERIFICATION_APPEND_FAILED
    if not rows:
        return

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    except OSError as exc:
        if not ROLLOUT_VERIFICATION_APPEND_FAILED:
            print(
                "⚠️ Failed to append rollout verification rows to "
                f"{output_path}: {exc}",
                flush=True,
            )
            ROLLOUT_VERIFICATION_APPEND_FAILED = True


def _trim_teacher_prompt_prefix(
    teacher_message_log: list[dict[str, Any]],
    max_total_sequence_length: int,
) -> None:
    total_tokens = sum(
        int(message["token_ids"].numel())
        for message in teacher_message_log
        if isinstance(message.get("token_ids"), torch.Tensor)
    )
    overflow = total_tokens - max_total_sequence_length
    if overflow <= 0:
        return

    for message in teacher_message_log:
        if overflow <= 0 or message.get("role") == "assistant":
            break
        token_ids = message.get("token_ids")
        if not isinstance(token_ids, torch.Tensor):
            continue
        trim = min(overflow, int(token_ids.numel()))
        if trim <= 0:
            continue
        message["token_ids"] = token_ids[trim:]
        token_loss_mask = message.get("token_loss_mask")
        if isinstance(token_loss_mask, torch.Tensor):
            message["token_loss_mask"] = token_loss_mask[trim:]
        overflow -= trim


def _build_teacher_rollout_message_logs(
    student_message_logs: list[Any],
    teacher_prompt_message_logs: list[Any],
    teacher_replace_message_counts: Optional[list[Any]],
    max_total_sequence_length: int,
) -> list[Any]:
    teacher_message_logs = []

    for sample_idx, student_message_log in enumerate(student_message_logs):
        teacher_prompt_message_log = (
            teacher_prompt_message_logs[sample_idx]
            if sample_idx < len(teacher_prompt_message_logs)
            else None
        )
        if not teacher_prompt_message_log:
            teacher_message_logs.append(deepcopy(student_message_log))
            continue

        replace_count = 0
        if (
            teacher_replace_message_counts is not None
            and sample_idx < len(teacher_replace_message_counts)
        ):
            try:
                replace_count = int(teacher_replace_message_counts[sample_idx])
            except (TypeError, ValueError):
                replace_count = 0
        replace_count = max(0, min(replace_count, len(student_message_log)))

        teacher_message_log = deepcopy(teacher_prompt_message_log) + deepcopy(
            student_message_log[replace_count:]
        )
        _apply_assistant_loss_mask([teacher_message_log])
        _trim_teacher_prompt_prefix(
            teacher_message_log,
            max_total_sequence_length=max_total_sequence_length,
        )
        teacher_message_logs.append(teacher_message_log)

    return teacher_message_logs


def _align_teacher_topk_to_student_masks(
    *,
    student_token_mask: torch.Tensor,
    teacher_token_mask: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    batch_size = int(student_token_mask.shape[0])
    student_seq_len = int(student_token_mask.shape[1])
    topk = int(teacher_topk_indices.shape[-1])

    aligned_logits = torch.zeros(
        (batch_size, student_seq_len, topk),
        dtype=teacher_topk_logits.dtype,
    )
    aligned_indices = torch.zeros(
        (batch_size, student_seq_len, topk),
        dtype=teacher_topk_indices.dtype,
    )
    invalid_samples: list[int] = []

    student_assistant_mask = student_token_mask[:, 1:].bool()
    teacher_assistant_mask = teacher_token_mask[:, 1:].bool()
    teacher_target_seq_len = int(teacher_assistant_mask.shape[1])

    teacher_logits_for_targets = teacher_topk_logits
    teacher_indices_for_targets = teacher_topk_indices
    teacher_topk_seq_len = int(teacher_topk_logits.shape[1])
    if teacher_topk_seq_len == teacher_target_seq_len + 1:
        teacher_logits_for_targets = teacher_topk_logits[:, :-1, :]
        teacher_indices_for_targets = teacher_topk_indices[:, :-1, :]
    elif teacher_topk_seq_len != teacher_target_seq_len:
        raise ValueError(
            "Teacher top-k sequence length does not match the teacher token mask. "
            f"Got top-k seq len {teacher_topk_seq_len} and teacher target mask len "
            f"{teacher_target_seq_len}."
        )

    for sample_idx in range(batch_size):
        student_positions = student_assistant_mask[sample_idx].nonzero(
            as_tuple=False
        ).squeeze(-1)
        teacher_positions = teacher_assistant_mask[sample_idx].nonzero(
            as_tuple=False
        ).squeeze(-1)

        student_count = int(student_positions.numel())
        teacher_count = int(teacher_positions.numel())
        if student_count == 0:
            continue
        if teacher_count != student_count:
            invalid_samples.append(sample_idx)
            continue

        aligned_logits[sample_idx, student_positions + 1] = teacher_logits_for_targets[
            sample_idx, teacher_positions
        ]
        aligned_indices[sample_idx, student_positions + 1] = (
            teacher_indices_for_targets[sample_idx, teacher_positions]
        )

    return aligned_logits, aligned_indices, invalid_samples


def distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: Optional[ColocatablePolicyInterface],
    student_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: DistillationLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: DistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """Run Distillation training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    # If student_generation is None, use the student_policy as the generation interface (megatron framework backend)
    if student_generation is None:
        student_generation = student_policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert student_generation is not None  # for mypy type check

    # common config/state items
    current_epoch = distillation_save_state["current_epoch"]  # current epoch
    current_step = distillation_save_state[
        "current_step"
    ]  # current step within current epoch
    total_steps = distillation_save_state[
        "total_steps"
    ]  # total number of steps across all epochs
    consumed_samples = distillation_save_state["consumed_samples"]
    total_valid_tokens = distillation_save_state["total_valid_tokens"]
    val_period = master_config["distillation"]["val_period"]
    val_at_start = master_config["distillation"]["val_at_start"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    generation_max_new_tokens = int(master_config["policy"]["generation"]["max_new_tokens"])
    teacher_update_mode = _get_teacher_update_mode(master_config["distillation"])
    teacher_update_period = _get_teacher_update_period(master_config["distillation"])
    use_reference_teacher = teacher_update_mode == "periodic_sync"
    rollout_greedy = bool(master_config["distillation"].get("rollout_greedy", False))
    max_epochs = master_config["distillation"][
        "max_num_epochs"
    ]  # max number of epochs to train for
    max_steps = master_config["distillation"][
        "max_num_steps"
    ]  # max number of steps to train for
    if use_reference_teacher and teacher_policy is not None:
        raise ValueError(
            "teacher_update_mode=periodic_sync expects the teacher to run from the "
            "student reference snapshot."
        )
    if not use_reference_teacher and teacher_policy is None:
        raise ValueError("A teacher policy is required when teacher_update_mode=fixed.")

    student_policy_with_reference = cast(Any, student_policy)
    warned_teacher_conditioned_topk = False
    warned_generation_cap = False
    rollout_verification_log_path = _get_rollout_verification_log_path()
    print(
        "▶ Rollout verification samples will be appended to "
        f"{rollout_verification_log_path}",
        flush=True,
    )

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...", flush=True)
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(
                student_policy, student_generation, colocated_inference
            )
            POLICY_GENERATION_STALE = False
        else:
            student_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            student_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=total_steps,
            master_config=master_config,
        )
        student_generation.finish_generation()
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    # Run distillation training (multi-epoch until reaching max_num_steps or max_num_epochs)
    batch: BatchedDataDict[DatumSpec]

    while total_steps < max_steps and current_epoch < max_epochs:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
            flush=True,
        )

        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(student_policy, total_steps + 1)
            if student_policy != student_generation:
                maybe_gpu_profile_step(student_generation, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                rollout_filter_metrics: dict[str, float] = {}
                sample_mask_before_rollout_filters: Optional[torch.Tensor] = None
                sample_mask_after_rollout_filters: Optional[torch.Tensor] = None

                # Prepare batch
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    # Repeat batch items
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            master_config["distillation"]["num_generations_per_prompt"]
                        )
                    )

                # Generate responses - this updates the LLMMessageLogType in repeated_batch
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            student_policy,
                            student_generation,
                            colocated_inference,
                            timer=timer,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()

                with timer.time("generation"):
                    # Use async rollouts if vLLM async engine is enabled
                    if _should_use_async_rollouts(master_config):
                        (
                            repeated_batch,
                            rollout_metrics,
                        ) = run_async_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["distillation"][
                                "max_rollout_turns"
                            ],
                            greedy=rollout_greedy,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["distillation"][
                                "max_rollout_turns"
                            ],
                            greedy=rollout_greedy,
                        )
                    student_generation.finish_generation()

                with timer.time("data_processing"):
                    # Add loss mask and advantages to each message in LLMMessageLogType
                    _apply_assistant_loss_mask(repeated_batch["message_log"])
                    sample_mask_before_rollout_filters = (
                        repeated_batch["loss_multiplier"].detach().clone().cpu()
                    )
                    rollout_filter_metrics = _filter_pathological_rollouts(
                        repeated_batch=repeated_batch,
                        tokenizer=tokenizer,
                        distillation_config=master_config["distillation"],
                        generation_max_new_tokens=generation_max_new_tokens,
                    )
                    sample_mask_after_rollout_filters = (
                        repeated_batch["loss_multiplier"].detach().clone().cpu()
                    )

                    # Convert updated LLMMessageLogType to FlatMessagesType for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data from flattened messages
                    train_data = BatchedDataDict[DistillationLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    # this will be mini-batched inside the policy, so maintain the packed multimodal structure
                    train_data.update(
                        flat_messages.get_multimodal_dict(as_tensors=False)
                    )
                    train_data.to("cpu")

                print("▶ Preparing for teacher logprob inference...", flush=True)
                with timer.time("teacher_logprob_inference_prep"):
                    if use_reference_teacher:
                        student_policy.prepare_for_lp_inference()
                    else:
                        teacher_policy.prepare_for_lp_inference()

                print("▶ Computing teacher logprobs...", flush=True)
                with timer.time("teacher_logprob_inference"):
                    teacher_prompt_message_logs = repeated_batch.get("teacher_message_log")
                    use_teacher_conditioned_prompts = bool(teacher_prompt_message_logs) and any(
                        teacher_prompt_message_log is not None
                        for teacher_prompt_message_log in teacher_prompt_message_logs
                    )

                    if use_teacher_conditioned_prompts:
                        if (
                            not master_config["loss_fn"].get("zero_outside_topk", False)
                            and not warned_teacher_conditioned_topk
                        ):
                            print(
                                "⚠️ Teacher-conditioned distillation is using "
                                "zero_outside_topk=false. Because teacher logits are "
                                "top-k only, probability mass outside the transmitted "
                                "support can drift and distort the KL target.",
                                flush=True,
                            )
                            warned_teacher_conditioned_topk = True
                        teacher_message_logs = _build_teacher_rollout_message_logs(
                            student_message_logs=repeated_batch["message_log"],
                            teacher_prompt_message_logs=teacher_prompt_message_logs,
                            teacher_replace_message_counts=repeated_batch.get(
                                "teacher_replace_message_count"
                            ),
                            max_total_sequence_length=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                        )
                        (
                            teacher_flat_messages,
                            teacher_input_lengths,
                        ) = batched_message_log_to_flat_message(
                            teacher_message_logs,
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                            make_sequence_length_divisible_by=master_config["policy"][
                                "make_sequence_length_divisible_by"
                            ],
                        )
                        teacher_data = BatchedDataDict[DistillationLossDataDict](
                            {
                                "input_ids": teacher_flat_messages["token_ids"],
                                "input_lengths": teacher_input_lengths,
                            }
                        )
                        teacher_data.update(
                            teacher_flat_messages.get_multimodal_dict(as_tensors=False)
                        )
                        teacher_data.to("cpu")

                        if use_reference_teacher:
                            teacher_topk = (
                                student_policy_with_reference.get_reference_topk_logits(
                                    teacher_data,
                                    k=master_config["distillation"]["topk_logits_k"],
                                )
                            )
                        else:
                            teacher_topk = teacher_policy.get_topk_logits(
                                teacher_data,
                                k=master_config["distillation"]["topk_logits_k"],
                            )
                        (
                            aligned_teacher_topk_logits,
                            aligned_teacher_topk_indices,
                            invalid_samples,
                        ) = _align_teacher_topk_to_student_masks(
                            student_token_mask=flat_messages["token_loss_mask"],
                            teacher_token_mask=teacher_flat_messages["token_loss_mask"],
                            teacher_topk_logits=teacher_topk["topk_logits"],
                            teacher_topk_indices=teacher_topk["topk_indices"],
                        )
                        train_data["teacher_topk_logits"] = aligned_teacher_topk_logits
                        train_data["teacher_topk_indices"] = aligned_teacher_topk_indices
                        if invalid_samples:
                            train_data["sample_mask"][invalid_samples] = 0
                            print(
                                "  ⚠️ Skipping "
                                f"{len(invalid_samples)} reference-conditioned samples "
                                "because teacher and student assistant token counts diverged.",
                                flush=True,
                            )
                    else:
                        if use_reference_teacher:
                            teacher_topk = (
                                student_policy_with_reference.get_reference_topk_logits(
                                    train_data,
                                    k=master_config["distillation"]["topk_logits_k"],
                                )
                            )
                        else:
                            teacher_topk = teacher_policy.get_topk_logits(
                                train_data,
                                k=master_config["distillation"]["topk_logits_k"],
                            )
                        train_data["teacher_topk_logits"] = teacher_topk["topk_logits"]
                        train_data["teacher_topk_indices"] = teacher_topk["topk_indices"]
                        invalid_samples = []

                rollout_verification_rows = _build_rollout_verification_rows(
                    step=total_steps,
                    tokenizer=tokenizer,
                    repeated_batch=repeated_batch,
                    distillation_config=master_config["distillation"],
                    generation_max_new_tokens=generation_max_new_tokens,
                    sample_mask_before_rollout_filters=sample_mask_before_rollout_filters,
                    sample_mask_after_rollout_filters=sample_mask_after_rollout_filters,
                    final_sample_mask=train_data.get("sample_mask"),
                    invalid_samples=invalid_samples,
                )
                _append_rollout_verification_rows(
                    rollout_verification_rows,
                    rollout_verification_log_path,
                )

                _log_debug_samples(
                    step=total_steps,
                    tokenizer=tokenizer,
                    train_data=train_data,
                    repeated_batch=repeated_batch,
                    invalid_samples=invalid_samples,
                )
                if (
                    generation_max_new_tokens > 0
                    and rollout_metrics["max_gen_tokens_per_sample"]
                    >= 0.9 * generation_max_new_tokens
                    and not warned_generation_cap
                ):
                    print(
                        "⚠️ On-policy rollouts are reaching the configured generation cap "
                        f"({rollout_metrics['max_gen_tokens_per_sample']:.0f}/"
                        f"{generation_max_new_tokens} tokens). Those truncated "
                        "continuations become distillation targets and often cause "
                        "loss/quality divergence. Lower policy.generation.max_new_tokens, "
                        "lower sampling temperature, or add stronger stop strings.",
                        flush=True,
                    )
                    warned_generation_cap = True

                teacher_sync_performed = 0.0
                valid_rollouts_after_filter = int(
                    (repeated_batch["loss_multiplier"] > 0).sum().item()
                )
                if valid_rollouts_after_filter == 0:
                    print(
                        "⚠️ All rollout targets were filtered before distillation. "
                        "Skipping the student update for this step.",
                        flush=True,
                    )
                    if teacher_policy is not None:
                        teacher_policy.offload_after_refit()
                    train_results = {
                        "loss": torch.tensor(0.0),
                        "grad_norm": torch.tensor(0.0),
                        "all_mb_metrics": {
                            "num_valid_samples": [0.0],
                            "global_valid_seqs": [0.0],
                            "global_valid_toks": [0.0],
                            "distill_clipped_tokens": [0.0],
                            "distill_loss_tokens": [0.0],
                        },
                    }
                else:
                    print("▶ Preparing for training...", flush=True)
                    with timer.time("training_prep"):
                        if teacher_policy is not None:
                            teacher_policy.offload_after_refit()
                        student_policy.prepare_for_training()  # set model train and reload optim to GPU
                        POLICY_GENERATION_STALE = True

                    print("▶ Training policy...", flush=True)
                    with timer.time("policy_training"):
                        train_results = student_policy.train(train_data, loss_fn)
                    if (
                        use_reference_teacher
                        and (total_steps + 1) % teacher_update_period == 0
                    ):
                        with timer.time("teacher_sync"):
                            student_policy_with_reference.sync_reference_model_from_current_model()
                        teacher_sync_performed = 1.0
                        print(
                            "▶ Synced reference teacher from student weights "
                            f"at step {total_steps + 1}.",
                            flush=True,
                        )

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # Run validation if it's a validation step
                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            student_policy, student_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        student_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                    )
                    student_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "mean_prompt_length": repeated_batch["length"].numpy(),
                    "total_num_tokens": input_lengths.numpy(),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {
                        "lr",
                        "wd",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                metrics.update(rollout_metrics)
                metrics.update(rollout_filter_metrics)
                metrics["teacher_sync_performed"] = teacher_sync_performed
                if metrics.get("distill_loss_tokens", 0.0) > 0:
                    metrics["distill_clip_fraction"] = (
                        metrics["distill_clipped_tokens"] / metrics["distill_loss_tokens"]
                    )
                if repeated_batch.size > 0:
                    metrics["filtered_rollouts_fraction"] = (
                        metrics["filtered_rollouts"] / float(repeated_batch.size)
                    )
                total_valid_tokens += metrics["global_valid_toks"]

                ## Checkpointing
                consumed_samples += master_config["distillation"][
                    "num_prompts_per_step"
                ]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                # +1 because total_steps is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()

                    distillation_save_state["current_epoch"] = current_epoch
                    distillation_save_state["current_step"] = current_step + 1
                    distillation_save_state["total_steps"] = total_steps + 1
                    distillation_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        distillation_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in distillation_save_state:
                        del distillation_save_state["val_reward"]
                    distillation_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_reward --> 'val:accuracy'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in distillation_save_state:
                                del distillation_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            distillation_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, distillation_save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            # Log training data
            log_data = {"content": flat_messages["content"]}
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            print("\n📊 Training Results:")

            print(f"  • Loss: {metrics['loss']:.4f}")
            print(
                f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
            )
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)",
                    flush=True,
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%",
                        flush=True,
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

            print("\n⏱️  Timing:", flush=True)
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            metrics.update(
                {
                    "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                    / total_time
                    / total_num_gpus
                }
            )

            print(f"  • Total step time: {total_time:.2f}s", flush=True)

            # Display all other timing metrics
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_steps:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        # End of epoch
        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation", flush=True)
        return {}, {}

    if val_task_to_env is None:
        print(
            "  ⚠️ No validation task to environment mapping provided, skipping validation",
            flush=True,
        )
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)

        total_rewards = []  # Can be any metric. Setted to 'accuracy' by default.
        total_lengths = []
        all_message_logs = []  # Collect all message logs

        max_val_samples = master_config["distillation"]["max_val_samples"]
        val_batch_size = master_config["distillation"]["val_batch_size"]
        max_batches: Optional[int] = None
        if max_val_samples > 0 and val_batch_size > 0:
            max_batches = max(1, math.ceil(max_val_samples / val_batch_size))

        for batch_idx, val_batch in enumerate(val_dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            # NeMo-Gym validation uses the HTTP-served vLLM endpoint and returns rewards
            # directly from the external environment.
            if "nemo_gym" in val_task_to_env:
                generation_config = deepcopy(master_config["policy"]["generation"])
                # NeMo-Gym validation goes through the OpenAI-compatible HTTP path,
                # so strip generation knobs that are only supported by direct vLLM use.
                generation_config["stop_strings"] = None
                generation_config["stop_token_ids"] = []
                generation_config["top_k"] = 0
                nemo_gym_rollout_result = run_async_nemo_gym_rollout(
                    policy_generation=policy_generation,
                    input_batch=val_batch,
                    tokenizer=tokenizer,
                    task_to_env=val_task_to_env,
                    max_seq_len=None,
                    generation_config=generation_config,
                    max_rollout_turns=None,
                    greedy=False,
                )
                val_batch = nemo_gym_rollout_result.final_batch
                gen_metrics = nemo_gym_rollout_result.rollout_metrics
            elif _should_use_async_rollouts(master_config):
                val_batch, gen_metrics = run_async_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["distillation"][
                        "max_rollout_turns"
                    ],
                    greedy=False,
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["distillation"][
                        "max_rollout_turns"
                    ],
                    greedy=False,
                )
            rewards = val_batch["total_reward"]

            total_rewards.extend(rewards.tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

            # Collect message logs for later display
            to_env = [
                get_keys_from_message_log(
                    val_batch["message_log"][i], ["role", "content"]
                )
                for i in range(len(val_batch["message_log"]))
            ]

            all_message_logs.extend(to_env)

        # Calculate validation metrics
        accuracy = (
            sum(total_rewards) / len(total_rewards) if len(total_rewards) > 0 else 0
        )
        avg_length = (
            sum(total_lengths) / len(total_lengths) if len(total_lengths) > 0 else 0
        )

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
        }

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...", flush=True)

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}", flush=True)

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
