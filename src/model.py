"""Bidirectional LSTM Encoder, a Luong/Bahdanau Attention Mechanism, and an LSTM Decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2, pad_idx=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # hidden: [num_layers * 2, batch_size, hidden_size]
        num_layers = hidden.size(0) // 2
        hidden_cat = torch.cat([hidden[0::2], hidden[1::2]], dim=2) # [num_layers, batch_size, hidden_size * 2]
        cell_cat = torch.cat([cell[0::2], cell[1::2]], dim=2)
        
        hidden = torch.tanh(self.fc_hidden(hidden_cat)) # [num_layers, batch_size, hidden_size]
        cell = torch.tanh(self.fc_cell(cell_cat))
        
        return outputs, (hidden, cell)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.U_a = nn.Linear(hidden_size * 2, hidden_size)
        self.V_a = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size * 2]
        src_len = encoder_outputs.size(1)
        
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1) # [batch_size, src_len, hidden_size]
        
        energy = torch.tanh(self.W_a(hidden_expanded) + self.U_a(encoder_outputs)) # [batch_size, src_len, hidden_size]
        scores = self.V_a(energy).squeeze(2) # [batch_size, src_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=1) # [batch_size, src_len]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) # [batch_size, 1, hidden_size * 2]
        
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2, pad_idx=0):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_size)
        
        # Input to LSTM is concatenated embedding + context vector
        self.lstm = nn.LSTM(
            embed_size + (hidden_size * 2),
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size + (hidden_size * 2) + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, hidden, cell, encoder_outputs, mask=None):
        # input_tok: [batch_size]
        input_tok = input_tok.unsqueeze(1) # [batch_size, 1]
        embedded = self.dropout(self.embedding(input_tok)) # [batch_size, 1, embed_size]
        
        # Query attention with top layer hidden state
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, mask) # [batch_size, 1, hidden_size * 2]
        
        lstm_input = torch.cat([embedded, context], dim=2) # [batch_size, 1, embed_size + hidden_size * 2]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell)) # output: [batch_size, 1, hidden_size]
        
        # Projection layer combining output, context, and embedding
        prediction_input = torch.cat([output, context, embedded], dim=2)
        prediction = self.fc_out(prediction_input.squeeze(1)) # [batch_size, vocab_size]
        
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx=0):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        mask = (src != self.pad_idx)
        input_tok = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder.forward_step(input_tok, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = output
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_tok = trg[:, t] if teacher_force else top1
            
        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, src, vocab, max_len=50, device='cpu'):
        self.eval()
        with torch.no_grad():
            encoder_outputs, (hidden, cell) = self.encoder(src)
            mask = (src != self.pad_idx)

            inputs = torch.tensor([vocab.sos_idx], device=device).unsqueeze(0)
            trg_indexes = [vocab.sos_idx]
            
            for _ in range(max_len):
                output, hidden, cell, _ = self.decoder.forward_step(
                    inputs, hidden, cell, encoder_outputs, mask
                )
                pred_token = output.argmax(1).item()
                if pred_token == vocab.eos_idx:
                    break
                trg_indexes.append(pred_token)
                inputs = torch.tensor([pred_token], device=device).unsqueeze(0)
                
        return [vocab.get_token(i) for i in trg_indexes[1:]]
