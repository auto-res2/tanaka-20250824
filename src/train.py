import os
import math
import time
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat
from entmax import sparsemax
import matplotlib.pyplot as plt

# ------------------------------
# Utilities
# ------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def available_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def savefig_pdf(path):
    plt.savefig(path, bbox_inches='tight')
    plt.close()


# ------------------------------
# Quantization wrappers (LSQ-lite + PACT)
# ------------------------------
class STEQuant(nn.Module):
    def __init__(self, bits=8, per_channel=False, ch_axis=-1, init_alpha=6.0):
        super().__init__()
        self.bits = bits
        self.qmin = - (2 ** (bits-1))
        self.qmax = (2 ** (bits-1)) - 1
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x, enable=True):
        if not enable:
            return x
        # PACT clipping
        x = torch.clamp(x, -self.alpha, self.alpha)
        # LSQ-ish scale estimation
        if self.per_channel:
            dim = self.ch_axis
            scale = (x.detach().abs().amax(dim=dim, keepdim=True) + 1e-6) / max(self.qmax, 1)
        else:
            scale = (x.detach().abs().amax() + 1e-6) / max(self.qmax, 1)
        # STE quantization
        x_q = torch.round(x / scale).clamp(self.qmin, self.qmax)
        x_hat = x_q * scale
        return x_hat


# ------------------------------
# Core: Token-to-Prototype Router (TPR)
# ------------------------------

def topk_prune(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    k = min(k, logits.size(-1))
    vals, idx = torch.topk(logits, k=k, dim=-1)
    return vals, idx


class TokenToPrototypeRouter(nn.Module):
    def __init__(self, d_model, num_heads, num_slots, k_top=4, use_dual_banks=True,
                 ema_decay=0.95, positional_bins=64, quant_path=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.h = num_heads
        self.S = num_slots
        self.k = k_top
        self.use_dual = use_dual_banks
        self.ema_decay = ema_decay
        self.pos_bins = positional_bins
        self.d_model = d_model

        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.slot_U = nn.Parameter(torch.randn(self.h, self.S, d_model // self.h) / math.sqrt(d_model // self.h))
        if use_dual_banks:
            self.pos_codes = nn.Parameter(torch.randn(self.h, self.S, d_model // self.h) / math.sqrt(d_model // self.h))
            self.bank_gate = nn.Linear(d_model, self.h)
        self.q_quant = STEQuant(bits=8)
        self.k_quant = STEQuant(bits=8)
        self.v_quant = STEQuant(bits=8)
        self.enable_quant = quant_path

    def _shape_heads(self, x):
        B, T, D = x.shape
        d = D // self.h
        return rearrange(x, 'b t (h d) -> b t h d', h=self.h), d

    def forward(self, x, causal_prefix=False, return_aux=False):
        # x: [B, T, D]
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        if self.enable_quant:
            q = self.q_quant(q)
            k = self.k_quant(k)
            v = self.v_quant(v)
        q, d = self._shape_heads(q)
        k, _ = self._shape_heads(k)
        v, _ = self._shape_heads(v)
        B, T, H, Dh = q.shape

        # token->slot logits using content bank
        logits = torch.einsum('b t h d, h s d -> b t h s', q, self.slot_U)
        if self.use_dual:
            pos_idx = torch.linspace(0, 1, steps=T, device=x.device)
            pos_emb = pos_idx[:, None] * torch.ones(Dh, device=x.device)[None, :]
            pos_emb = repeat(pos_emb, 't d -> b t h d', b=B, h=H)
            pos_logits = torch.einsum('b t h d, h s d -> b t h s', pos_emb, self.pos_codes)
            gate = torch.sigmoid(self.bank_gate(rearrange(q.mean(dim=1), 'b h d -> b (h d)')))
            gate = rearrange(gate, 'b h -> b h 1 1')
            logits = gate * logits + (1 - gate) * pos_logits

        # top-k then sparsemax
        topk_vals, topk_idx = topk_prune(logits, self.k)
        gates_sparse = sparsemax(topk_vals, dim=-1)
        gates_full = torch.zeros_like(logits)
        gates_full.scatter_(-1, topk_idx, gates_sparse)

        # prototype accumulation
        mass = gates_full  # [B,T,H,S]
        Kv = torch.einsum('b t h s, b t h d -> b t h s d', mass, k)
        Vv = torch.einsum('b t h s, b t h d -> b t h s d', mass, v)
        mass_cum = mass.cumsum(dim=1) if causal_prefix else mass.sum(dim=1, keepdim=True)
        K_cum = Kv.cumsum(dim=1) if causal_prefix else Kv.sum(dim=1, keepdim=True)
        V_cum = Vv.cumsum(dim=1) if causal_prefix else Vv.sum(dim=1, keepdim=True)
        eps = 1e-6
        K_proto = K_cum / (mass_cum.unsqueeze(-1) + eps)
        V_proto = V_cum / (mass_cum.unsqueeze(-1) + eps)

        if causal_prefix:
            K_out = K_proto  # [B,T,H,S,Dh]
            V_out = V_proto
        else:
            K_out = K_proto[:, -1]  # [B,H,S,Dh]
            V_out = V_proto[:, -1]

        aux = {'gates': gates_full, 'mass': mass, 'q': q}
        return (K_out, V_out, q), (aux if return_aux else None)


# ------------------------------
# Sparse Hopfield Memory Retrieval (S-HMR)
# ------------------------------
class HopfieldRetriever(nn.Module):
    def __init__(self, num_heads, beta_init=1.0, quant_path=True):
        super().__init__()
        self.h = num_heads
        self.beta = nn.Parameter(torch.full((num_heads,), float(beta_init)))
        self.enable_quant = quant_path
        self.o_quant = STEQuant(bits=8)

    def forward(self, q, K_proto, V_proto, out_proj: nn.Linear):
        # q: [B,T,H,Dh]; K/V proto: [B,T?,H,S,Dh] or [B,H,S,Dh]
        if K_proto.dim() == 4:
            # Expand per time step
            B, H, S, Dh = K_proto.shape
            T = q.size(1)
            K = K_proto[:, None, ...].expand(B, T, H, S, Dh)
            V = V_proto[:, None, ...].expand_as(K)
        else:
            K = K_proto
            V = V_proto
        beta = rearrange(self.beta, 'h -> 1 1 h 1')
        scores = torch.einsum('b t h d, b t h s d -> b t h s', q, K) * beta
        attn = sparsemax(scores, dim=-1)
        y = torch.einsum('b t h s, b t h s d -> b t h d', attn, V)
        y = rearrange(y, 'b t h d -> b t (h d)')
        if self.enable_quant:
            y = self.o_quant(y)
        return out_proj(y), attn


# ------------------------------
# Blocks and Tiny Models
# ------------------------------
class ProtoHopBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, S=32, k_top=4, ffn_hidden=256, quant_path=True):
        super().__init__()
        self.router = TokenToPrototypeRouter(d_model, n_heads, S, k_top=k_top, quant_path=quant_path)
        self.retriever = HopfieldRetriever(n_heads, quant_path=quant_path)
        self.out_proj = nn.Linear(d_model, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )

    def forward(self, x):
        h = self.pre_norm(x)
        (Kp, Vp, q), aux = self.router(h, causal_prefix=True, return_aux=True)
        y, attn = self.retriever(q, Kp, Vp, self.out_proj)
        x = x + y
        x = x + self.ffn(x)
        return x, {'attn_proto': attn, **(aux or {})}


class MHSA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.h)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.h)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.h)
        attn_scores = torch.einsum('b h t d, b h s d -> b h t s', q, k) / math.sqrt(self.dh)
        # causal mask
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask.bool().unsqueeze(0), float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        y = torch.einsum('b h t s, b h s d -> b h t d', attn, v)
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.o_proj(y)


class BaselineBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, ffn_hidden=256):
        super().__init__()
        self.attn = MHSA(d_model, n_heads)
        self.pre_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )

    def forward(self, x):
        h = self.pre_norm(x)
        y = self.attn(h)
        x = x + y
        x = x + self.ffn(x)
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, n_layers=2, n_heads=4, ffn_hidden=256, proto=False, S=32, k_top=4):
        super().__init__()
        self.proto = proto
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        if proto:
            for _ in range(n_layers):
                self.layers.append(ProtoHopBlock(d_model, n_heads, S=S, k_top=k_top, ffn_hidden=ffn_hidden))
        else:
            for _ in range(n_layers):
                self.layers.append(BaselineBlock(d_model, n_heads, ffn_hidden))
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        aux_last = None
        for layer in self.layers:
            if self.proto:
                x, aux_last = layer(x)
            else:
                x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, aux_last


# ------------------------------
# Cross-Attention modules (Experiment 2)
# ------------------------------
class MHCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, tgt, src):
        q = rearrange(self.q_proj(tgt), 'b t (h d) -> b h t d', h=self.h)
        k = rearrange(self.k_proj(src), 'b s (h d) -> b h s d', h=self.h)
        v = rearrange(self.v_proj(src), 'b s (h d) -> b h s d', h=self.h)
        attn_scores = torch.einsum('b h t d, b h s d -> b h t s', q, k) / math.sqrt(self.dh)
        attn = F.softmax(attn_scores, dim=-1)
        y = torch.einsum('b h t s, b h s d -> b h t d', attn, v)
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.o_proj(y), attn


class ProtoCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, S=32, k_top=4, use_dual=True):
        super().__init__()
        self.h = n_heads
        self.router_src = TokenToPrototypeRouter(d_model, n_heads, S, k_top=k_top, use_dual_banks=use_dual)
        self.retriever = HopfieldRetriever(n_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def build_source_prototypes(self, src_hidden, chunk_size=128):
        B, T, D = src_hidden.shape
        chunks = torch.split(src_hidden, chunk_size, dim=1)
        K_list, V_list = [], []
        for c in chunks:
            (Kp, Vp, _), _ = self.router_src(c, causal_prefix=False, return_aux=False)
            K_list.append(Kp)
            V_list.append(Vp)
        K_src = torch.cat(K_list, dim=2)  # [B,H,S*num_chunks,Dh]
        V_src = torch.cat(V_list, dim=2)
        return K_src, V_src

    def forward(self, tgt_hidden, src_prototypes):
        K_src, V_src = src_prototypes
        q = rearrange(tgt_hidden, 'b t (h d) -> b t h d', h=self.retriever.h)
        out, attn = self.retriever(q, K_src, V_src, self.out_proj)
        return out, attn


# ------------------------------
# CSLRP (Experiment 3)
# ------------------------------
class CSLRPLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True, groups=8, r=4, C=128, M=1):
        super().__init__()
        assert d_in % groups == 0 and d_out % groups == 0
        self.din_g = d_in // groups
        self.dout_g = d_out // groups
        self.groups = groups
        self.r = r
        self.M = M
        self.E_in = nn.Parameter(torch.randn(C, self.din_g, r) / math.sqrt(self.din_g))
        self.E_out = nn.Parameter(torch.randn(C, self.dout_g, r) / math.sqrt(self.dout_g))
        self.idx_in = nn.Parameter(torch.randn(M, groups, C))
        self.idx_out = nn.Parameter(torch.randn(M, groups, C))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.act_quant = STEQuant(bits=8)
        self.w_quant = STEQuant(bits=8, per_channel=True, ch_axis=-1)

    def _one_hot_indices(self, logits):
        y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        return y  # [M,G,C]

    def forward(self, x, quant=True):
        B = x.size(0)
        if quant:
            x = self.act_quant(x)
        xg = x.view(B, self.groups, self.din_g)
        sel_in = self._one_hot_indices(self.idx_in)     # [M,G,C]
        sel_out = self._one_hot_indices(self.idx_out)   # [M,G,C]
        U = torch.einsum('mgc, cdr -> mgdr', sel_in, self.E_in)   # [M,G,din_g,r]
        V = torch.einsum('mgc, cdr -> mgdr', sel_out, self.E_out) # [M,G,dout_g,r]
        t = torch.einsum('mgdr, bgd -> mbgr', U, xg)              # [M,B,G,r]
        yg = torch.einsum('mgdr, mbgr -> bgd', V, t)              # [B,G,dout_g]
        y = yg.reshape(B, -1)
        if self.bias is not None:
            y = y + self.bias
        if quant:
            y = self.w_quant(y)
        return y


# ------------------------------
# Synthetic Datasets
# ------------------------------
class SyntheticLMDataset(Dataset):
    def __init__(self, vocab_size=256, seq_len=256, size=1024, pattern='bigram'):
        self.vocab = vocab_size
        self.seq_len = seq_len
        self.size = size
        self.pattern = pattern

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.pattern == 'bigram':
            x = torch.randint(0, self.vocab, (self.seq_len,))
            y = torch.roll(x, shifts=-1)
        elif self.pattern == 'needle':
            x = torch.randint(0, self.vocab, (self.seq_len,))
            needle_len = max(3, self.seq_len // 32)
            start = random.randint(0, self.seq_len - 2*needle_len - 1)
            needle = torch.randint(0, self.vocab, (needle_len,))
            x[start:start+needle_len] = needle
            x[-needle_len-1:-1] = needle
            y = torch.roll(x, shifts=-1)
        elif self.pattern == 'kvcopy':
            x = torch.randint(0, self.vocab, (self.seq_len,))
            kv_table = {k: 16 + (k % 16) for k in range(16)}
            y = torch.zeros_like(x)
            last_key = None
            for t in range(self.seq_len-1):
                if x[t] < 16:
                    last_key = int(x[t].item())
                y[t] = kv_table[last_key] if last_key is not None else x[t]
            y[-1] = 0
        else:
            x = torch.randint(0, self.vocab, (self.seq_len,))
            y = torch.roll(x, shifts=-1)
        return x.long(), y.long()


class SyntheticSummarizationDataset(Dataset):
    def __init__(self, vocab_size=512, src_len=1024, tgt_len=64, size=256, pattern='every_k'):
        self.vocab = vocab_size
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.size = size
        self.pattern = pattern

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        src = torch.randint(0, self.vocab, (self.src_len,))
        if self.pattern == 'every_k':
            k = max(1, self.src_len // self.tgt_len)
            pos = torch.arange(0, self.src_len, k)[:self.tgt_len]
        else:
            seg = self.src_len // self.tgt_len
            pos = []
            for i in range(self.tgt_len):
                s = i*seg
                e = min((i+1)*seg, self.src_len)
                window = src[s:e]
                j = int(torch.argmax(window).item())
                pos.append(s+j)
            pos = torch.tensor(pos)
        tgt = src[pos]
        return src.long(), tgt.long(), pos.long()


class SyntheticClassification(Dataset):
    def __init__(self, d_in=64, num_classes=16, size=512, patterns=('linear', 'xor')):
        self.d_in = d_in
        self.num_classes = num_classes
        self.size = size
        self.patterns = patterns

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        p = random.choice(self.patterns)
        x = torch.randn(self.d_in)
        if p == 'linear':
            W = torch.arange(1, self.d_in+1).float()
            y = int(((x * W).sum() > 0).item()) % self.num_classes
        else:
            y = int(((x[:2] > 0).int().sum() % 4).item())
        return x, y


# ------------------------------
# Loss/metric helpers and plotting
# ------------------------------

def train_lm_one_epoch(model, loader, optimizer, device, log_every=0):
    model.train()
    total_loss = 0.0
    losses = []
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), y[:, :-1].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        losses.append(float(loss.item()))
        if log_every and (i+1) % log_every == 0:
            print(f"  [LM] Step {i+1}/{len(loader)} loss={loss.item():.4f}")
    return total_loss / max(1, len(loader)), losses


@torch.no_grad()
def eval_perplexity_lm(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), y[:, :-1].reshape(-1), reduction='sum')
        total_loss += float(loss.item())
        total_tokens += (y[:, :-1].numel())
    ppl = math.exp(total_loss / max(1, total_tokens))
    return ppl


def plot_series(series_list: List[List[float]], labels: List[str], title: str, ylabel: str, out_path: str):
    plt.figure(figsize=(5,4))
    for s, lb in zip(series_list, labels):
        plt.plot(s, label=lb)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    savefig_pdf(out_path)


def plot_latency(results, fname):
    Ls = [r[0] for r in results]
    proto = [r[1] for r in results]
    dense = [r[2] for r in results]
    plt.figure(figsize=(5,4))
    plt.plot(Ls, proto, marker='o', label='ProtoHop')
    plt.plot(Ls, dense, marker='s', label='Dense')
    plt.xlabel('Context length L')
    plt.ylabel('ms/token (lower is better)')
    plt.title('Decode latency vs context length')
    plt.legend()
    plt.grid(True)
    savefig_pdf(fname)


def plot_bars(categories, values_a, values_b, labels, ylabel, title, fname):
    x = np.arange(len(categories))
    w = 0.35
    plt.figure(figsize=(6,4))
    plt.bar(x-w/2, values_a, width=w, label=labels[0])
    plt.bar(x+w/2, values_b, width=w, label=labels[1])
    plt.xticks(x, categories)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis='y')
    savefig_pdf(fname)


def confusion_matrix(pred_bins: List[int], true_bins: List[int], num_bins: int) -> np.ndarray:
    cm = np.zeros((num_bins, num_bins), dtype=np.int64)
    for p, t in zip(pred_bins, true_bins):
        if 0 <= p < num_bins and 0 <= t < num_bins:
            cm[t, p] += 1
    return cm


def plot_confusion(cm: np.ndarray, fname: str, title: str = 'Confusion matrix', xlabel='Pred bin', ylabel='True bin'):
    plt.figure(figsize=(4.5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks_x = np.arange(cm.shape[1])
    tick_marks_y = np.arange(cm.shape[0])
    plt.xticks(tick_marks_x)
    plt.yticks(tick_marks_y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    savefig_pdf(fname)


# ------------------------------
# Experiment 1: Streaming LM with Prototype KV-Cache Compression
# ------------------------------

def kv_cache_memory_bytes(L: int, S: int, H: int, Dh: int, dtype_bytes: int = 4):
    dense = L * H * Dh * 2 * dtype_bytes
    proto = S * H * Dh * 2 * dtype_bytes
    return dense, proto


def measure_decode_latency(model_proto: TinyDecoderLM, model_dense: TinyDecoderLM, device, seq_lengths=(256, 512, 1024), warmup=2, steps=8):
    print("Measuring decode-like per-token latency (synthetic)...")
    results = []
    for L in seq_lengths:
        B = 1
        x = torch.randint(0, 256, (B, L), device=device)
        for _ in range(warmup):
            _ = model_proto(x)
            _ = model_dense(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            _ = model_proto(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms_proto = (t1 - t0) * 1000.0 / steps / L

        t0 = time.perf_counter()
        for _ in range(steps):
            _ = model_dense(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms_dense = (t1 - t0) * 1000.0 / steps / L

        results.append((L, ms_proto, ms_dense))
        print(f"  L={L}: ProtoHop {ms_proto:.4f} ms/tok, Dense {ms_dense:.4f} ms/tok")
    return results


def run_experiment1(config: Dict, image_dir: str, models_dir: str, device=None) -> Dict:
    print("\n=== Experiment 1 — Streaming Long-Context LM with Prototype KV-Cache Compression ===")
    device = device or available_device()
    set_seed(int(config.get('seed', 42)))
    ensure_dir(image_dir)
    ensure_dir(models_dir)

    vocab_size = int(config.get('vocab_size', 256))
    d_model = int(config.get('d_model', 128))
    n_layers = int(config.get('n_layers', 2))
    n_heads = int(config.get('n_heads', 4))
    ffn_hidden = int(config.get('ffn_hidden', 256))
    S = int(config.get('S', 32))
    k_top = int(config.get('k_top', 4))
    batch_size = int(config.get('batch_size', 8))
    seq_len = int(config.get('seq_len', 256))
    train_size = int(config.get('train_size', 64))

    ds_bigram = SyntheticLMDataset(vocab_size=vocab_size, seq_len=seq_len, size=train_size, pattern='bigram')
    ds_needle = SyntheticLMDataset(vocab_size=vocab_size, seq_len=seq_len, size=train_size, pattern='needle')
    ds_kvcopy = SyntheticLMDataset(vocab_size=vocab_size, seq_len=seq_len, size=train_size, pattern='kvcopy')

    loader_bigram = DataLoader(ds_bigram, batch_size=batch_size, shuffle=True)
    loader_needle = DataLoader(ds_needle, batch_size=batch_size, shuffle=False)
    loader_kvcopy = DataLoader(ds_kvcopy, batch_size=batch_size, shuffle=False)

    model_proto = TinyDecoderLM(vocab_size, d_model, n_layers, n_heads, ffn_hidden, proto=True, S=S, k_top=k_top).to(device)
    model_dense = TinyDecoderLM(vocab_size, d_model, n_layers, n_heads, ffn_hidden, proto=False).to(device)
    opt_proto = torch.optim.AdamW(model_proto.parameters(), lr=float(config.get('lr', 1e-3)))
    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=float(config.get('lr', 1e-3)))

    print("Training on bigram pattern (proto vs dense)...")
    log_every = max(1, len(loader_bigram)//2)
    loss_proto_bigram, losses_proto_bigram = train_lm_one_epoch(model_proto, loader_bigram, opt_proto, device, log_every)
    loss_dense_bigram, losses_dense_bigram = train_lm_one_epoch(model_dense, loader_bigram, opt_dense, device, log_every)

    ppl_proto_bigram = eval_perplexity_lm(model_proto, loader_bigram, device)
    ppl_dense_bigram = eval_perplexity_lm(model_dense, loader_bigram, device)
    ppl_proto_needle = eval_perplexity_lm(model_proto, loader_needle, device)
    ppl_dense_needle = eval_perplexity_lm(model_dense, loader_needle, device)
    ppl_proto_kvcopy = eval_perplexity_lm(model_proto, loader_kvcopy, device)
    ppl_dense_kvcopy = eval_perplexity_lm(model_dense, loader_kvcopy, device)

    plot_series([losses_proto_bigram, losses_dense_bigram], ["ProtoHop", "Dense"],
                title="Training loss", ylabel="Loss",
                out_path=os.path.join(image_dir, "training_loss_protohop_vs_baseline.pdf"))

    results = measure_decode_latency(
        model_proto, model_dense, device,
        seq_lengths=tuple(config.get('latency_contexts', [256, 512, 1024])),
        warmup=int(config.get('latency_warmup', 1)),
        steps=int(config.get('latency_steps', 4))
    )
    plot_latency(results, fname=os.path.join(image_dir, "inference_latency_protohop_vs_baseline.pdf"))

    H, Dh = n_heads, d_model // n_heads
    lengths = list(config.get('latency_contexts', [256, 512, 1024]))
    dense_bytes = []
    proto_bytes = []
    for L in lengths:
        d_b, p_b = kv_cache_memory_bytes(L=L, S=S, H=H, Dh=Dh, dtype_bytes=4)
        dense_bytes.append(d_b/(1024**2))
        proto_bytes.append(p_b/(1024**2))
    plot_bars([str(L) for L in lengths], proto_bytes, dense_bytes,
              labels=("ProtoHop-KV", "Dense-KV"),
              ylabel='Memory (MiB)',
              title='KV cache memory vs context length',
              fname=os.path.join(image_dir, "kv_memory_protohop_vs_baseline.pdf"))

    # Save checkpoints
    proto_ckpt = os.path.join(models_dir, 'exp1_protohop.pt')
    dense_ckpt = os.path.join(models_dir, 'exp1_dense.pt')
    torch.save({'state_dict': model_proto.state_dict(), 'config': dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, ffn_hidden=ffn_hidden, S=S, k_top=k_top)}, proto_ckpt)
    torch.save({'state_dict': model_dense.state_dict(), 'config': dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, ffn_hidden=ffn_hidden)}, dense_ckpt)

    print(f"Perplexity (bigram): ProtoHop={ppl_proto_bigram:.2f} vs Dense={ppl_dense_bigram:.2f}")
    print(f"Perplexity (needle):  ProtoHop={ppl_proto_needle:.2f} vs Dense={ppl_dense_needle:.2f}")
    print(f"Perplexity (kvcopy):  ProtoHop={ppl_proto_kvcopy:.2f} vs Dense={ppl_dense_kvcopy:.2f}")

    return {
        'ppl': {
            'bigram': (ppl_proto_bigram, ppl_dense_bigram),
            'needle': (ppl_proto_needle, ppl_dense_needle),
            'kvcopy': (ppl_proto_kvcopy, ppl_dense_kvcopy),
        },
        'latency_results': results,
        'checkpoints': {'proto': proto_ckpt, 'dense': dense_ckpt}
    }


# ------------------------------
# Experiment 2: Cross-Attention with Monotonic Alignment
# ------------------------------
class MonotonicPrior(nn.Module):
    def __init__(self, skip_lambda=0.01, back_lambda=0.1):
        super().__init__()
        self.skip_lambda = skip_lambda
        self.back_lambda = back_lambda

    def forward(self, attn_proto, proto_pos):
        # attn_proto: [B, T_dec, H, S]
        exp_pos = torch.einsum('b t h s, s -> b t h', attn_proto, proto_pos)
        deltas = exp_pos[:, 1:] - exp_pos[:, :-1]
        back = F.relu(-deltas).mean()
        skip = deltas.pow(2).mean()
        return self.back_lambda*back + self.skip_lambda*skip


def run_experiment2(config: Dict, image_dir: str, models_dir: str, device=None) -> Dict:
    print("\n=== Experiment 2 — Causal Cross-Attention with Monotonic Prototype Alignment ===")
    device = device or available_device()
    set_seed(int(config.get('seed', 123)))
    ensure_dir(image_dir)
    ensure_dir(models_dir)

    vocab_size = int(config.get('vocab_size', 512))
    d_model = int(config.get('d_model', 128))
    n_heads = int(config.get('n_heads', 4))
    S = int(config.get('S', 32))
    chunk_size = int(config.get('chunk_size', 128))
    batch_size = int(config.get('batch_size', 4))
    src_len = int(config.get('src_len', 1024))
    tgt_len = int(config.get('tgt_len', 64))
    size = int(config.get('train_size', 64))

    ds_everyk = SyntheticSummarizationDataset(vocab_size=vocab_size, src_len=src_len, tgt_len=tgt_len, size=size, pattern='every_k')
    ds_heads = SyntheticSummarizationDataset(vocab_size=vocab_size, src_len=src_len, tgt_len=tgt_len, size=size, pattern='headlines')
    loader_everyk = DataLoader(ds_everyk, batch_size=batch_size, shuffle=True)
    loader_heads = DataLoader(ds_heads, batch_size=batch_size, shuffle=True)

    src_emb = nn.Embedding(vocab_size, d_model).to(device)
    tgt_emb = nn.Embedding(vocab_size, d_model).to(device)
    proto_xattn = ProtoCrossAttention(d_model=d_model, n_heads=n_heads, S=S, k_top=4, use_dual=True).to(device)
    dense_xattn = MHCrossAttention(d_model=d_model, num_heads=n_heads).to(device)

    mono_prior = MonotonicPrior(skip_lambda=float(config.get('skip_lambda', 0.01)), back_lambda=float(config.get('back_lambda', 0.1)))

    opt_proto = torch.optim.AdamW(list(src_emb.parameters())+list(tgt_emb.parameters())+list(proto_xattn.parameters()), lr=float(config.get('lr', 1e-3)))
    opt_dense = torch.optim.AdamW(list(src_emb.parameters())+list(tgt_emb.parameters())+list(dense_xattn.parameters()), lr=float(config.get('lr', 1e-3)))

    def train_one_epoch_cross(loader, use_proto=True):
        if use_proto:
            proto_xattn.train()
        else:
            dense_xattn.train()
        total = 0.0
        losses = []
        for i, (src, tgt, pos) in enumerate(loader):
            src = src.to(device)
            tgt = tgt.to(device)
            pos = pos.to(device)
            opt = opt_proto if use_proto else opt_dense
            opt.zero_grad()

            src_h = src_emb(src)
            tgt_h = tgt_emb(tgt)

            if use_proto:
                K_src, V_src = proto_xattn.build_source_prototypes(src_h, chunk_size=chunk_size)
                out, attn = proto_xattn(tgt_h, (K_src, V_src))
                num_chunks = math.ceil(src_h.size(1)/chunk_size)
                Sbins = S * num_chunks
                true_bins = torch.clamp((pos // chunk_size) * S + (pos % chunk_size) * S // chunk_size, 0, Sbins-1)
                attn_avg = attn.mean(dim=2)
                loss_ce = F.cross_entropy(attn_avg.reshape(-1, attn_avg.size(-1)), true_bins.reshape(-1))
                proto_pos = torch.linspace(0, 1, steps=Sbins, device=device)
                loss_mono = mono_prior(attn_avg.unsqueeze(2), proto_pos)
                loss = loss_ce + float(config.get('mono_weight', 0.1))*loss_mono
            else:
                out, attn = dense_xattn(tgt_h, src_h)
                num_chunks = math.ceil(src_h.size(1)/chunk_size)
                Sbins = S * num_chunks
                attn_avg = attn.mean(dim=1)
                # Pool attention into S bins per chunk
                pooled_rows = []
                for b in range(attn_avg.size(0)):
                    row_t = []
                    for t in range(attn_avg.size(1)):
                        v = attn_avg[b, t]
                        bins = []
                        for ch in range(num_chunks):
                            s = ch*chunk_size
                            e = min((ch+1)*chunk_size, src_h.size(1))
                            seg = v[s:e]
                            sub_bins = torch.split(seg, max(1, (e-s)//S))
                            vals = torch.stack([sb.mean() for sb in sub_bins])
                            if vals.size(0) < S:
                                vals = torch.cat([vals, vals.new_zeros(S - vals.size(0))])
                            vals = vals[:S]
                            bins.append(vals)
                        row_t.append(torch.cat(bins))
                    pooled_rows.append(torch.stack(row_t))
                attn_bins = torch.stack(pooled_rows).to(device)
                true_bins = torch.clamp((pos // chunk_size) * S + (pos % chunk_size) * S // chunk_size, 0, Sbins-1)
                loss = F.cross_entropy(attn_bins.reshape(-1, attn_bins.size(-1)), true_bins.reshape(-1))

            loss.backward()
            opt.step()
            total += float(loss.item())
            losses.append(float(loss.item()))
            if (i+1) % max(1, len(loader)//2) == 0:
                print(f"  [XATTN] Step {i+1}/{len(loader)} loss={loss.item():.4f}")
        return total / max(1, len(loader)), losses

    print("Training ProtoCrossAttention on 'every_k'...")
    proto_loss_everyk, proto_losses_everyk = train_one_epoch_cross(loader_everyk, use_proto=True)
    print("Training Dense CrossAttention on 'every_k'...")
    dense_loss_everyk, dense_losses_everyk = train_one_epoch_cross(loader_everyk, use_proto=False)

    plot_series([proto_losses_everyk, dense_losses_everyk], ["ProtoHop Cross", "Dense Cross"],
                title="Cross-attention training loss", ylabel="Loss",
                out_path=os.path.join(image_dir, "training_loss_crossattention_protohop_vs_dense.pdf"))

    # Confusion matrix on 'headlines'
    print("Evaluating alignment on 'headlines' pattern (confusion matrix)...")
    proto_xattn.eval(); dense_xattn.eval()
    pred_bins_proto = []
    true_bins_all = []
    with torch.no_grad():
        for src, tgt, pos in loader_heads:
            src = src.to(device); tgt = tgt.to(device); pos = pos.to(device)
            src_h = src_emb(src); tgt_h = tgt_emb(tgt)
            K_src, V_src = proto_xattn.build_source_prototypes(src_h, chunk_size=chunk_size)
            _, attn = proto_xattn(tgt_h, (K_src, V_src))
            num_chunks = math.ceil(src_h.size(1)/chunk_size)
            Sbins = S * num_chunks
            attn_avg = attn.mean(dim=2)
            pb = torch.argmax(attn_avg, dim=-1)
            tb = torch.clamp((pos // chunk_size) * S + (pos % chunk_size) * S // chunk_size, 0, Sbins-1)
            pred_bins_proto += pb.reshape(-1).tolist()
            true_bins_all += tb.reshape(-1).tolist()
    cm = confusion_matrix(pred_bins_proto, true_bins_all, num_bins=S * math.ceil(src_len/chunk_size))
    plot_confusion(cm, fname=os.path.join(image_dir, "confusion_matrix_crossalignment_protohop.pdf"),
                   title='ProtoHop cross-attn alignment')

    # Attention entropy on every_k
    def attention_entropy_stats(use_proto=True):
        entropies = []
        with torch.no_grad():
            for src, tgt, _ in loader_everyk:
                src = src.to(device); tgt = tgt.to(device)
                src_h = src_emb(src); tgt_h = tgt_emb(tgt)
                if use_proto:
                    K_src, V_src = proto_xattn.build_source_prototypes(src_h, chunk_size=chunk_size)
                    _, attn = proto_xattn(tgt_h, (K_src, V_src))
                    p = attn.mean(dim=2) + 1e-8
                else:
                    _, attn = dense_xattn(tgt_h, src_h)
                    p = attn.mean(dim=1) + 1e-8
                ent = (-(p * p.log()).sum(dim=-1)).mean()
                entropies.append(float(ent))
        return np.mean(entropies)

    ent_proto = attention_entropy_stats(True)
    ent_dense = attention_entropy_stats(False)
    plot_bars(['every_k'], [ent_proto], [ent_dense], labels=("ProtoHop", "Dense"),
              ylabel='Attention entropy (nats)', title='Cross-attention entropy',
              fname=os.path.join(image_dir, "attention_entropy_crossattention_protohop_vs_dense.pdf"))

    # Save minimal checkpoints (embeddings + modules)
    torch.save({'src_emb': src_emb.state_dict(), 'tgt_emb': tgt_emb.state_dict(), 'proto_xattn': proto_xattn.state_dict(), 'config': dict(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, S=S, chunk_size=chunk_size)},
               os.path.join(models_dir, 'exp2_protohop_cross.pt'))
    torch.save({'src_emb': src_emb.state_dict(), 'tgt_emb': tgt_emb.state_dict(), 'dense_xattn': dense_xattn.state_dict(), 'config': dict(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads)},
               os.path.join(models_dir, 'exp2_dense_cross.pt'))

    return {
        'train_loss_everyk': (proto_loss_everyk, dense_loss_everyk),
        'entropy_everyk': (ent_proto, ent_dense)
    }


# ------------------------------
# Experiment 3: Quantization-Native via CSLRP
# ------------------------------
class TinyClassifier(nn.Module):
    def __init__(self, d_in=64, d_hidden=128, num_classes=16, use_cslrp=False):
        super().__init__()
        if use_cslrp:
            self.fc1 = CSLRPLinear(d_in, d_hidden, groups=8, r=4, C=128, M=1)
            self.fc2 = CSLRPLinear(d_hidden, num_classes, groups=8, r=4, C=128, M=1)
        else:
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.fc2 = nn.Linear(d_hidden, num_classes)
        self.act = nn.ReLU()
        self.use_cslrp = use_cslrp

    def forward(self, x):
        h = self.act(self.fc1(x.float()))
        logits = self.fc2(h)
        return logits


def run_experiment3(config: Dict, image_dir: str, models_dir: str, device=None) -> Dict:
    print("\n=== Experiment 3 — Quantization-Native Training via CSLRP ===")
    device = device or available_device()
    set_seed(int(config.get('seed', 7)))
    ensure_dir(image_dir)
    ensure_dir(models_dir)

    d_in = int(config.get('d_in', 64))
    d_hidden = int(config.get('d_hidden', 128))
    num_classes = int(config.get('num_classes', 16))
    size = int(config.get('train_size', 256))
    batch_size = int(config.get('batch_size', 32))

    ds = SyntheticClassification(d_in=d_in, num_classes=num_classes, size=size, patterns=('linear','xor'))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model_cslrp = TinyClassifier(d_in=d_in, d_hidden=d_hidden, num_classes=num_classes, use_cslrp=True).to(device)
    model_dense = TinyClassifier(d_in=d_in, d_hidden=d_hidden, num_classes=num_classes, use_cslrp=False).to(device)

    opt_cslrp = torch.optim.AdamW(model_cslrp.parameters(), lr=float(config.get('lr', 2e-3)))
    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=float(config.get('lr', 2e-3)))

    def train_one_epoch(model, opt):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        return np.mean(losses), losses

    print("Training CSLRP classifier...")
    cslrp_loss, cslrp_losses = train_one_epoch(model_cslrp, opt_cslrp)
    print("Training Dense classifier (baseline)...")
    dense_loss, dense_losses = train_one_epoch(model_dense, opt_dense)

    plot_series([cslrp_losses, dense_losses], ["CSLRP", "Dense-Linear"],
                title="Classifier training loss", ylabel="Loss",
                out_path=os.path.join(image_dir, "training_loss_cslrp_vs_linear.pdf"))

    def outlier_rate(model):
        x = torch.randn(128, d_in, device=device)
        with torch.no_grad():
            if isinstance(model.fc1, CSLRPLinear):
                a = model.fc1.act_quant.alpha.item()
                h = model.fc1.act_quant(model.fc1.act_quant(x))
                rate = float((h.abs() > a).float().mean().item())
            else:
                h = model.fc1(x)
                a = float((h.std().item() * 3.0))
                rate = float((h.abs() > a).float().mean().item())
        return rate

    out_cslrp = outlier_rate(model_cslrp)
    out_dense = outlier_rate(model_dense)

    plot_bars(['probe'], [out_cslrp], [out_dense], labels=("CSLRP", "Dense-Linear"),
              ylabel='Outlier rate (>|alpha|)', title='Activation outliers (lower is better)',
              fname=os.path.join(image_dir, "activation_outliers_cslrp_vs_linear.pdf"))

    @torch.no_grad()
    def eval_acc(model):
        model.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        return correct/ max(1,total)

    acc_cslrp = eval_acc(model_cslrp)
    acc_dense = eval_acc(model_dense)
    print(f"Accuracy: CSLRP={acc_cslrp:.3f} vs Dense-Linear={acc_dense:.3f}")

    # Save checkpoints
    torch.save({'state_dict': model_cslrp.state_dict(), 'config': dict(d_in=d_in, d_hidden=d_hidden, num_classes=num_classes)}, os.path.join(models_dir, 'exp3_cslrp.pt'))
    torch.save({'state_dict': model_dense.state_dict(), 'config': dict(d_in=d_in, d_hidden=d_hidden, num_classes=num_classes)}, os.path.join(models_dir, 'exp3_dense.pt'))

    return {
        'train_loss': (cslrp_loss, dense_loss),
        'acc': (acc_cslrp, acc_dense),
        'outlier_rate': (out_cslrp, out_dense)
    }


# ------------------------------
# Public entrypoints used by main/evaluate
# ------------------------------

def train_all_experiments(cfg: Dict) -> Dict:
    device = available_device()
    image_dir = cfg.get('image_dir', '.research/iteration1/images')
    models_dir = cfg.get('models_dir', 'models')
    ensure_dir(image_dir)
    ensure_dir(models_dir)

    results = {}
    if cfg.get('run_exp1', True):
        results['exp1'] = run_experiment1(cfg.get('exp1', {}), image_dir, models_dir, device)
    if cfg.get('run_exp2', True):
        results['exp2'] = run_experiment2(cfg.get('exp2', {}), image_dir, models_dir, device)
    if cfg.get('run_exp3', True):
        results['exp3'] = run_experiment3(cfg.get('exp3', {}), image_dir, models_dir, device)
    return results
