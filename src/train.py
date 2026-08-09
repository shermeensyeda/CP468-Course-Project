"""Sets Seed Reproducibility, Tracks Model Size, Monitors Total Training Time, and Exports Checkpoints"""

import os
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.vocabulary import Vocabulary
from src.dataset import GECDataset, make_collate_fn  # Adjust import based on dataset.py class name
from src.model import Encoder, Decoder, Seq2Seq

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        src = batch['source'].to(device)
        trg = batch['target'].to(device)
        
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            
            output = model(src, trg, teacher_forcing_ratio=0.0) # Turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=None, help='limit training set size for quick runs')
    args = parser.parse_args()

    SET_SEED = 42
    set_seed(SET_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Vocabulary
    vocab = Vocabulary.load("data/processed/vocab.json")

    # Initialize Datasets & DataLoaders
    train_dataset = GECDataset("data/processed/train.jsonl", vocab)
    val_dataset = GECDataset("data/processed/validation.jsonl", vocab)

    if args.limit:
        train_dataset = Subset(train_dataset, list(range(min(args.limit, len(train_dataset)))))

    PAD_IDX = vocab.pad_idx
    collate = make_collate_fn(PAD_IDX)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Model Hyperparameters
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3

    enc = Encoder(len(vocab.itos), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, PAD_IDX)
    dec = Decoder(len(vocab.itos), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, PAD_IDX)
    model = Seq2Seq(enc, dec, PAD_IDX).to(device)

    print(f"Total Trainable Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    os.makedirs("checkpoints", exist_ok=True)

    N_EPOCHS = args.epochs
    CLIP = 1.0
    best_val_loss = float('inf')

    start_time = time.time()
    for epoch in range(N_EPOCHS):
        e_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        e_end = time.time()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_lstm.pt")

        # also save last checkpoint
        torch.save(model.state_dict(), "checkpoints/last_lstm.pt")

        print(f"Epoch {epoch+1:02} | Time: {e_end - e_start:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in: {total_time / 60:.2f} minutes")

    # Save metadata for report writing
    meta = {
        "parameters": model.count_parameters(),
        "training_time_seconds": total_time,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "best_val_loss": best_val_loss
    }
    with open("checkpoints/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
