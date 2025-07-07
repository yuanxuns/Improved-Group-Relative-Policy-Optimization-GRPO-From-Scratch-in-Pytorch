import torch
import torch.nn as nn
from pathlib import Path

import gc
from collections import defaultdict
import numpy as np
import dataclasses
import math

from src.models.qwen.qwen2_model import Transformer
from src.models.grpo.memory_efficient_adam import MemoryEfficientAdamW
from src.dataset.countdown_ds import Instance, MiniBatch
from typing import Callable, List
from src.tokenizers.tokenizer import Tokenizer


def build_and_load_model(config, device: torch.device, build_optimzier: bool = True):
    """
    Builds a model and optimizer from a config dictionary.

    Args:
        config: A dictionary with configuration options.
        device: torch.device.
        build_optimizer: Whether to build an optimizer. Defaults to True.

    Returns:
        A tuple containing the model and optimizer. The optimizer is None if build_optimizer is False.
    """
    model = Transformer.from_pretrained(
        config["model"]["pretrained_model_path"], device=device
    )

    if config["model"]["preload_ckpt_file"] is not None:
        file_path = (
            Path(config["training"]["ckpt_dir"]) / config["model"]["preload_ckpt_file"]
        )
        if file_path.exists():
            print(f"Loading model weights from {file_path}")
            model.load_state_dict(torch.load(file_path, weights_only=True))
    else:
        print(
            f"Preload checkpoint file {config['model']['preload_ckpt_file']} does not exist, skipping loading."
        )

    optimizer = None
    if build_optimzier:
        optimizer = MemoryEfficientAdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            betas=config["training"]["betas"],
            enabled=config["training"]["memory_efficient_adamw"],
        )

    return model, optimizer


@torch.no_grad()
def single_step_rollout(
    model: Transformer,
    ref_model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    config,
) -> List[Instance]:
    """
    Perform a single step rollout for a batch of questions.

    Args:
        model: The model to use for the rollout.
        ref_model: The reference model to use for the rollout.
        batch: A MiniBatch of questions and context.
        tokenizer: The tokenizer to use for the rollout.
        max_gen_len: The maximum length to generate.
        num_answer_per_question: The number of answers to generate for each question.
        reward_function: A function to compute the reward for each generated answer.
        device: The device to use for the rollout.
        dtype: The dtype to use for the rollout.
        config: The configuration dictionary.

    Returns:
        A list of Instance objects with the generated info.
    """
    ref_model.to(device)
    ref_model.eval()

    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    # print("end_token_id:", end_token_id)
    # print("pad_token_id:", pad_token_id)

    prefix_token_ids = batch.prefix_token_ids
    question_size = len(batch.prefix_text)
    batch_size = question_size * num_answer_per_question
    min_prefix_token_len = min(len(t) for t in prefix_token_ids)
    max_prefix_token_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prefix_token_len
    model.init_kv_cache(
        max_batch_size=batch_size,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )

    # (B, total_len)
    tokens = torch.full(
        (batch_size, total_len), pad_token_id, dtype=torch.long, device=device
    )

    # fills the prefix tokens.
    for k, t in enumerate(prefix_token_ids):
        start = k * num_answer_per_question
        end = start + num_answer_per_question
        tokens[start:end, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    prev_pos = 0
    input_text_mask = tokens != pad_token_id  # (B, total_len)
    is_finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prefix_token_len, total_len):
        print(
            f"\r* Generating tokens: {cur_pos-min_prefix_token_len:>4d}/{total_len-min_prefix_token_len:>4d}",
            flush=True,
            end="",
        )
        with torch.no_grad():
            # Scales it by temperature, (B, tgt_len, vocab_size)
            logits = (
                model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
                / config["training"]["temperature"]
            )
        # (B, tgt_len, vocab_size) -> (B, vocab_size)
        probs = torch.softmax(logits[:, -1], dim=-1)
        # (B, 1)
        next_token = torch.multinomial(probs, num_samples=1)
        # (B, 1) -> (B)
        next_token = next_token.reshape(-1)
        # (B)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id, (B)
        next_token = torch.where(is_finished, pad_token_id, next_token)
        # updates the tokens tensor with the next token, (B, total_len)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_pad_token = next_token == pad_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = (
                is_finished
                | (is_end_token & is_generated_token)
                | (is_pad_token & is_generated_token)
            )
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # prepare the output instances
    instances = []
    for i in range(question_size):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = tokenizer.detokenize(generated_token_ids)
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
                config=config,
            )

            with torch.no_grad():
                batch_token_ids = torch.tensor(
                    [batch.prefix_token_ids[i] + generated_token_ids],
                    device=device,
                    dtype=torch.int64,
                )
                old_per_token_log_probs = get_per_token_logps_and_entropies(
                    model,
                    batch_token_ids,
                    pad_token_id,
                    compute_entropies=False,
                    temperature=config["training"]["temperature"],
                )["per_token_logps"]
                ref_per_token_log_probs = get_per_token_logps_and_entropies(
                    ref_model,
                    batch_token_ids,
                    pad_token_id,
                    compute_entropies=False,
                    temperature=config["training"]["temperature"],
                )["per_token_logps"]

            instance = Instance(
                prefix_text=batch.prefix_text[i],
                full_text=batch.prefix_text[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
                old_per_token_log_probs=old_per_token_log_probs,
                ref_per_token_log_probs=ref_per_token_log_probs,
            )
            instances.append(instance)
    # clear the output line
    print("\r", end=" " * 100, flush=True)
    ref_model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return instances


def normalize_rewards_per_group(instances: List[Instance]) -> List[Instance]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for instance in instances:
        groups[tuple(instance.prefix_text)].append(instance)
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for instance in group:
            normalized_reward = (instance.reward - mean_reward) / (std_reward + 1e-6)
            instance = dataclasses.replace(instance, reward=normalized_reward)
            output.append(instance)
    return output


def get_per_token_entropies_from_logits(logits, chunk_size: int = 1) -> torch.Tensor:
    """
    Compute the Shannon entropy for each row of *logits* without
    materialising the full soft-max in memory.
    The batch dimension is processed in chunks of size `chunk_size` so that
    only a subset of rows is expanded to probabilities at any one time.
    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all
            leading dimensions are preserved.
        chunk_size (`int`, *optional*, defaults to `1`):
            Number of rows to process per iteration.
    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    per_token_entropies = []
    for logits_chunk in logits.split(chunk_size, dim=0):
        logps = torch.nn.functional.log_softmax(logits_chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        per_token_entropies.extend(chunk_entropy)

    per_token_entropies = torch.stack(per_token_entropies)
    return per_token_entropies


def get_per_token_logps_and_entropies(
    model,
    batch_token_ids,
    pad_token_id,
    compute_logps: bool = True,
    compute_entropies: bool = True,
    temperature: float = 1.0,
):
    """
    Compute the per-token log-probabilities and/or entropies of the given batch.

    Args:
        model: The model to use for computing the logits.
        batch_token_ids: The input batch of token ids.
        pad_token_id: The token id of the padding token.
        compute_logps: Whether to compute the per-token log-probabilities.
        compute_entropies: Whether to compute the per-token entropies.
        temperature: The temperature to use for scaling the logits.

    Returns:
        A dictionary containing the per-token log-probabilities and/or entropies.
    """
    per_token_logps = None
    per_token_entropies = None

    # (B, batch_len-1, vocab_size)
    logits = model.forward(batch_token_ids[:, :-1]) / temperature

    if compute_entropies:
        with torch.no_grad():
            # (B, batch_len-1)
            per_token_entropies = get_per_token_entropies_from_logits(logits)

    if compute_logps:
        # (B * (batch_len-1)) -> (B, batch_len-1)
        per_token_logps = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # (B * (batch_len-1), vocab_size)
            batch_token_ids[:, 1:].reshape(-1),  # (B * (batch_len-1))
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(batch_token_ids.shape[0], -1)

    return {
        "per_token_logps": per_token_logps,
        "per_token_entropies": per_token_entropies,
    }


def update_policy(
    model,
    optimizer,
    instances: List[Instance],
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    config,
):
    """
    Update the policy using the given instances.

    Args:
        model: The policy model to update.
        optimizer: The optimizer to use for updating the policy.
        instances: The list of instances to use for updating the policy.
        pad_token_id: The id of the padding token.
        device: The device to use for updating the policy.
        dtype: The data type to use for updating the policy.
        config: The configuration to use for updating the policy.

    Returns:
        A dictionary containing the loss, gradient norm, and entropy.
    """

    if len(instances) == 0:
        print("No instances to update policy.")
        return {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0}

    instances = normalize_rewards_per_group(instances)

    num_target_tokens = sum(len(instance.generated_token_ids) for instance in instances)
    entropy = 0.0
    micro_batch_size = config["training"]["micro_batch_size"]

    for i in range(0, len(instances), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(instances):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(instances))
        batch_instances = instances[i:j]
        batch_lengths = [
            len(instance.prefix_token_ids) + len(instance.generated_token_ids)
            for instance in batch_instances
        ]
        batch_max_length = max(batch_lengths)

        # (microB, batch_max_length)
        batch_token_ids = torch.tensor(
            [
                instance.prefix_token_ids
                + instance.generated_token_ids
                + [pad_token_id] * (batch_max_length - batch_lengths[i])
                for i, instance in enumerate(batch_instances)
            ],
            device=device,
            dtype=torch.int64,
        )

        # (microB, batch_max_length - 1)
        batch_old_per_token_log_probs = torch.cat(
            [
                torch.nn.functional.pad(
                    instance.old_per_token_log_probs,
                    (
                        0,
                        batch_max_length - 1 - instance.old_per_token_log_probs.size(1),
                    ),
                    value=0.0,
                    mode="constant",
                )
                for instance in batch_instances
            ],
            dim=0,
        ).to(device=device, dtype=dtype)

        # (microB, batch_max_length - 1)
        batch_ref_per_token_log_probs = torch.cat(
            [
                torch.nn.functional.pad(
                    instance.ref_per_token_log_probs,
                    (
                        0,
                        batch_max_length - 1 - instance.ref_per_token_log_probs.size(1),
                    ),
                    value=0.0,
                    mode="constant",
                )
                for instance in batch_instances
            ],
            dim=0,
        ).to(device=device, dtype=dtype)

        # (microB, batch_max_length)
        batch_masks = torch.tensor(
            [
                [0] * len(instance.prefix_token_ids)
                + [1] * len(instance.generated_token_ids)
                + [0] * (batch_max_length - batch_lengths[i])
                for i, instance in enumerate(batch_instances)
            ],
            device=device,
            dtype=torch.bool,
        )

        # (microB)
        batch_advantages = torch.tensor(
            [instance.reward for instance in batch_instances],
            device=device,
            dtype=dtype,
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            result = get_per_token_logps_and_entropies(
                model,
                batch_token_ids,
                pad_token_id,
                temperature=config["training"]["temperature"],
            )
        # (microB, batch_max_length-1)
        per_token_logps = result["per_token_logps"]
        # (microB, batch_max_length-1)
        per_token_entropies = result["per_token_entropies"]

        entropy_threshold = torch.quantile(
            per_token_entropies.flatten().float(),
            config["training"]["token_entropy_percentile_threshold"],
        )
        entropy_mask = per_token_entropies >= entropy_threshold

        # (microB, batch_max_length-1)
        target_masks = batch_masks[:, 1:]

        entropy = (
            entropy + (per_token_entropies * target_masks).sum() / num_target_tokens
        )

        # (microB, batch_max_length-1)
        cur_old_policy_ratio = torch.exp(
            per_token_logps - batch_old_per_token_log_probs
        )
        per_token_loss1 = cur_old_policy_ratio * batch_advantages[:, None]
        per_token_loss2 = (
            torch.clamp(
                cur_old_policy_ratio,
                1 - config["training"]["eps_low"],
                1 + config["training"]["eps_high"],
            )
            * batch_advantages[:, None]
        )

        # (microB, batch_max_length-1)
        ref_cur_policy_ratio = torch.exp(
            batch_ref_per_token_log_probs - per_token_logps
        )
        per_token_kl = (
            ref_cur_policy_ratio - (batch_ref_per_token_log_probs - per_token_logps) - 1
        )
        per_token_loss = (
            torch.min(per_token_loss1, per_token_loss2)
            - config["training"]["beta"] * per_token_kl
        )
        per_token_loss = per_token_loss * target_masks * entropy_mask
        loss = -per_token_loss.sum() / num_target_tokens

        loss.backward()

    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=config["training"]["max_grad_norm"]
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }
