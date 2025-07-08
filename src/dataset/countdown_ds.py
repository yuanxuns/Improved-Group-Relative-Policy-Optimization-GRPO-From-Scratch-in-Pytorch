from click import prompt
from torch.utils.data import Dataset
from src.tokenizers.tokenizer import Tokenizer
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import torch

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first reason about the given task, then "
    "provide the user with an answer."
)
USER_TEMPLATE = (
    "Using the provided information, answer the question. "
    "Given a list of three or four numbers {numbers} and the target {target}, "
    "you use the numbers exactly once to create an equation whose value equals "
    "the target. Feel free to change the order of numbers if needed, and "
    "use arithmetic operations: brackets (), addition +, subtraction -, multiplication * and division. "
    "Show your reasoning processes and verifies the answer in <think></think>, then "
    "return a correct equation in <answer> </answer>."
)
RESPONSE_PROMPT = (
    "Let me try different equation combinations to obtain a correct answer. "
    "I will solve this step by step. <think>"
)


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    numbers: List[List[int]]
    target: List[int]
    prefix_text: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]


@dataclass
class Instance:
    """Store all relevant information of an instance."""

    prefix_text: str
    full_text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]
    old_per_token_log_probs: torch.Tensor
    ref_per_token_log_probs: torch.Tensor


class CountdownDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        split: str = "train",
        test_size: int = 100,
    ) -> None:
        data = pd.read_parquet(data_path)
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(self.tokenize_prefix(item["nums"], item["target"]))
        return item

    def tokenize_prefix(self, nums: List[int], target: int) -> Dict[str, Any]:
        user_message = USER_TEMPLATE.format(numbers=nums, target=target)
        reponse_prompt = RESPONSE_PROMPT.format(numbers=nums, target=target)
        prefix_text = self.tokenizer.encode_chat_with_response_prompt(
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            prompt=reponse_prompt,
        )
        tokens = self.tokenizer.tokenize(prefix_text)
        return {
            "prefix_text": prefix_text,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }


def collate_fn(batch_info: List[Dict[str, Any]]) -> MiniBatch:
    numbers = [element["nums"] for element in batch_info]
    target = [element["target"] for element in batch_info]
    prefix_text = [element["prefix_text"] for element in batch_info]
    prefix_tokens = [element["prefix_tokens"] for element in batch_info]
    prefix_token_ids = [element["prefix_token_ids"] for element in batch_info]
    return MiniBatch(
        numbers=numbers,
        target=target,
        prefix_text=prefix_text,
        prefix_tokens=prefix_tokens,
        prefix_token_ids=prefix_token_ids,
    )
