import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size, maxlen, rpe):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        pe = torch.zeros(maxlen, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, maxlen).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.norm = nn.LayerNorm(d_model)
        self.rpe = rpe

    def forward(self, x):
        if self.rpe:
            embedding = self.tok_embed(x)
        else:
            embedding = self.tok_embed(x) + self.pe[:, :x.size(1)]
        return self.norm(embedding)

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen, rpe):
        super().__init__()
        assert d_model % nhead == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(drop)
        self.resid_dropout = nn.Dropout(drop)
        self.register_buffer("bias", torch.tril(torch.ones(maxlen, maxlen)).view(1, 1, maxlen, maxlen))
        self.rpe = rpe
        rpe = torch.zeros(1, nhead, maxlen, maxlen)
        for i in range(1, maxlen):
            rpe = rpe - torch.tril(torch.ones(maxlen, maxlen), diagonal=-i).view(1, 1, maxlen, maxlen)
        for i in range(nhead):
            rpe[0, i] = rpe[0, i] * 2 **(-8 / nhead * (i + 1))
        self.register_buffer("RPE", rpe)
        self.n_head = nhead
        self.n_embd = d_model
        

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.rpe:
            att = att + self.RPE[:, :, :T, :T]
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen, rpe):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, nhead=nhead, drop=drop, maxlen=maxlen, rpe=rpe)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(d_model, 4 * d_model),
            c_proj  = nn.Linear(4 * d_model, d_model),
            act     = NewGELU(),
            dropout = nn.Dropout(drop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            embedding = Embedding(d_model=args.dmodel, vocab_size=args.vocab, maxlen=args.maxlen, rpe=args.rpe),
            drop = nn.Dropout(args.drop),
            h = nn.ModuleList([Block(d_model=args.dmodel, nhead=args.head, drop=args.drop, maxlen=args.maxlen, rpe=args.rpe) for _ in range(args.num_layer)]),
            ln_f = nn.LayerNorm(args.dmodel),
        ))
        self.lm_head = nn.Linear(args.dmodel, args.vocab, bias=True)

    def forward(self, idx):
        b, t = idx.size()
        emb = self.transformer.embedding(idx)
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
        
    def generate(self, idx, start):
        b, t = idx.size()
        tmp_start = start + 0
        while True:
            logits = self.forward(idx)
            idx_new = torch.argmax(logits, dim=2)
            idx[torch.arange(b), tmp_start + 1] = idx_new[torch.arange(b), tmp_start]
            if (torch.sum(idx_new[torch.arange(b), tmp_start] != 2) == 0) or (torch.sum(tmp_start == t - 2) != 0):
                break
            tmp_start[idx_new[torch.arange(b), tmp_start] != 2] += 1
        return idx