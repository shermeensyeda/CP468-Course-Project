"""Training-only word vocabulary with explicit special tokens."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

class Vocabulary:
    def __init__(self, min_freq: int = 2, max_size: int | None = 30000):
        self.min_freq=min_freq
        self.max_size=max_size
        self.itos=list(SPECIAL_TOKENS)
        self.stoi={tok:i for i,tok in enumerate(self.itos)}

    def build(self, token_sequences):
        counts=Counter(tok for seq in token_sequences for tok in seq)
        items=[(tok,c) for tok,c in counts.items() if c >= self.min_freq and tok not in self.stoi]
        items.sort(key=lambda x:(-x[1],x[0]))
        if self.max_size is not None:
            items=items[:max(0,self.max_size-len(self.itos))]
        for tok,_ in items:
            self.stoi[tok]=len(self.itos)
            self.itos.append(tok)
        return self

    @property
    def pad_idx(self): return self.stoi["<PAD>"]
    @property
    def unk_idx(self): return self.stoi["<UNK>"]
    @property
    def sos_idx(self): return self.stoi["<SOS>"]
    @property
    def eos_idx(self): return self.stoi["<EOS>"]

    def encode(self, tokens, add_boundaries=True):
        ids=[self.stoi.get(t,self.unk_idx) for t in tokens]
        return ([self.sos_idx]+ids+[self.eos_idx]) if add_boundaries else ids

    def decode(self, ids, skip_special=True):
        toks=[]
        for i in ids:
            tok=self.itos[i] if 0 <= i < len(self.itos) else "<UNK>"
            if skip_special and tok in SPECIAL_TOKENS: continue
            toks.append(tok)
        return toks

    def save(self,path):
        Path(path).write_text(json.dumps({"min_freq":self.min_freq,"max_size":self.max_size,"itos":self.itos},indent=2,ensure_ascii=False),encoding='utf-8')

    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_text(encoding='utf-8'))
        v=cls(obj['min_freq'],obj['max_size']); v.itos=obj['itos']; v.stoi={t:i for i,t in enumerate(v.itos)}; return v
