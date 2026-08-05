"""PyTorch dataset and dynamic-padding collate function."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .text_utils import tokenize
from .vocabulary import Vocabulary

class GECDataset(Dataset):
    def __init__(self, jsonl_path, vocab: Vocabulary, max_length: int = 60):
        self.rows=[json.loads(line) for line in Path(jsonl_path).read_text(encoding='utf-8').splitlines() if line.strip()]
        self.vocab=vocab; self.max_length=max_length
    def __len__(self): return len(self.rows)
    def _encode(self,text):
        toks=tokenize(text)[:self.max_length-2]
        return torch.tensor(self.vocab.encode(toks),dtype=torch.long)
    def __getitem__(self,idx):
        row=self.rows[idx]
        return {"source":self._encode(row['source']),"target":self._encode(row['target']),"id":row['id']}

def make_collate_fn(pad_idx: int):
    def collate(batch):
        src=pad_sequence([b['source'] for b in batch],batch_first=True,padding_value=pad_idx)
        tgt=pad_sequence([b['target'] for b in batch],batch_first=True,padding_value=pad_idx)
        return {"source":src,"target":tgt,"source_mask":src.ne(pad_idx),"target_mask":tgt.ne(pad_idx),"ids":[b['id'] for b in batch]}
    return collate
