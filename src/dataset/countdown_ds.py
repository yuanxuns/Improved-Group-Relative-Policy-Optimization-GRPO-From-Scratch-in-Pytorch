from click import prompt
from torch.utils.data import Dataset
from src.tokenizers.tokenizer import Tokenizer
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

SYSTEM_MESSAGE = (
  "You are a helpful assistant. You first reason about the given task, then "
  "provide the user with an answer."
)
USER_TEMPLATE = (
  "Using the provided information, answer the question. "
  "Given a list of numbers {numbers} and the target {target}, you can use the limited "
  "arithmetic operations +, -, *, /, (, ) and numbers to "
  "create an equation that equals the target. Each given number must be used exactly once, and feel free to change the order of numbers if needed. "
  "You can only selectively use the provided arithmetic operations. You do not have to use all provided "
  " arithmetic operations, and feel free to use any given arithmetic operations multiple times. "
  "Show your reasoning processes in the <think></think> tags. "
  "The reasoning steps should keep trying multiple attempts until the equation equals the target. Finally, return the correct equation "
  "in <answer> </answer> tags. For example, "
  "suppose given a list of numbers [3 1 6 2] and the target 12. <think>Given numbers [3 1 6 2] and the target 12, I will try to find an equation that equals the target. " 
  "The first try, 3 * 6 + 1 - 2 = 17, which is not equal to the target 12. Continue. "
  "The second try,  6 + (2 * 3 - 1) = 11, which is not equal to the target 12. Continue. "
  "The third try, (1 + 3) / 2 * 6, which is equal to the target. Succeed and return the equation."
  "</think> "
  "<answer>(1 + 3) / 2 * 6</answer>."
)
RESPONSE_PROMPT = "Given a list of numbers {numbers} and the target {target}. Let me solve this step by step.<think>"

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
    

class CountdownDataset(Dataset):
    def __init__(self, data_path:str, tokenizer:Tokenizer,
        split: str = "train",
        test_size: int = 100,)->None:
      data = pd.read_parquet(data_path)
      self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
      )      
      self.tokenizer = tokenizer
      
    def __len__(self)->int:
      return len(self.data)
    
    def __getitem__(self, idx):
      item = self.data.iloc[idx].to_dict()
      item.update(self.tokenize_prefix(item["nums"], item["target"]))
      return item
    
    def tokenize_prefix(self, nums:List[int], target: int)->Dict[str, Any]:
      user_message = USER_TEMPLATE.format(numbers=nums, target=target)
      prefix_text = self.tokenizer.encode_chat_with_response_prompt(
        messages = [
          {"role": "system", "content": SYSTEM_MESSAGE},
          {"role": "user", "content": user_message}
        ],
        prompt = RESPONSE_PROMPT,
      )
      tokens = self.tokenizer.tokenize(prefix_text)
      return {
        "prefix_text": prefix_text,
        "prefix_tokens": tokens.tokens,
        "prefix_token_ids": tokens.ids,
      }
      
def collate_fn(batch_info: List[Dict[str, Any]])->MiniBatch:
  numbers = [element["nums"] for element in batch_info]
  target = [element["target"] for element in batch_info]
  prefix_text = [element["prefix_text"] for element in batch_info]
  prefix_tokens = [element["prefix_tokens"] for element in batch_info]
  prefix_token_ids = [element["prefix_token_ids"] for element in batch_info]
  return MiniBatch(
    numbers = numbers,
    target=target,
    prefix_text=prefix_text,
    prefix_tokens=prefix_tokens,
    prefix_token_ids=prefix_token_ids,
  )
  