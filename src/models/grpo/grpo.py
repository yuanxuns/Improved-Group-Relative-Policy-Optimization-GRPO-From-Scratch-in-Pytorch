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

def build_and_load_model(config, device:torch.device):

    model = Transformer.from_pretrained(config["model"]["pretrained_model_path"], device=device)
    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    if config["model"]["preload_ckpt_file"] is not None:
        file_path = Path(config["training"]["ckpt_dir"]) / config["model"]["preload_ckpt_file"]
        if file_path.exists():
            print(f"Loading model weights from {file_path}")
            model.load_state_dict(
                torch.load(file_path, weights_only=True)
            )
        
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
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    config,
) -> List[Instance]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
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
    tokens = torch.full((batch_size, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )
    # prefix_token_ids_tensor = [torch.tensor(t, dtype=torch.long, device=device) for t in prefix_token_ids]
    # for k, t_tensor in enumerate(prefix_token_ids_tensor):
    #     start = k * num_answer_per_question
    #     end = start + num_answer_per_question
    #     tokens[start:end, : t_tensor.size(0)] = t_tensor
    
    prev_pos = 0
    # (B, total_len)
    input_text_mask = tokens != pad_token_id
    assert min_prefix_token_len < total_len
    is_finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prefix_token_len, total_len):
        print(
            f"\r* Generating tokens: {cur_pos-min_prefix_token_len:>4d}/{total_len-min_prefix_token_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        # (B, tgt_len, vocab_size) -> (B, 1)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        # (B, 1) -> (B)
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token)
        # (B, total_len)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
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
            instance = Instance(
                prefix_text=batch.prefix_text[i],
                full_text=batch.prefix_text[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            instances.append(instance)
    # clear the output line
    print("\r", end=" " * 100, flush=True)
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

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    The formula logsumexp(logits) - sum(probs * logits) is equivalent to the definition of Shannon entropy, H(P) = - sum(P * log(P)), where P is the probability distribution obtained from softmax.
    Recall that probs = exp(logits) / sum(exp(logits)).
    Taking the logarithm of probs: log(probs) = logits - log(sum(exp(logits))).
    The term log(sum(exp(logits))) is equivalent to logsumexp(logits).
    Substituting into the entropy formula: H(P) = - sum(P * (logits - logsumexp(logits))).
    H(P) = - sum(P * logits) + sum(P * logsumexp(logits)).
    Since the sum of probabilities is 1 (sum(P) = 1), the second term simplifies to logsumexp(logits).
    Thus, H(P) = logsumexp(logits) - sum(P * logits). 
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy
    
def update_policy(
    model,
    optimizer,
    instances: List[Instance],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Update the policy using the GRPO algorithm."""
    instances = normalize_rewards_per_group(instances)
    # sort instances by token length for efficient (micro-)batching
    instances.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(instances) / micro_batch_size)
    num_target_tokens = sum(len(instance.generated_token_ids) for instance in instances)
    entropy = 0.0

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
        batch_token_ids = [
            instance.prefix_token_ids
            + instance.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, instance in enumerate(batch_instances)
        ]
        batch_masks = [
            [0] * len(instance.prefix_token_ids)
            + [1] * len(instance.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, instance in enumerate(batch_instances)
        ]
        batch_advantages = [instance.reward for instance in batch_instances]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            logits = model.forward(input_token_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        obj = log_probs * batch_advantages[:, None]
        # per-token objective
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }

    