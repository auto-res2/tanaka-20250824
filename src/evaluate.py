import os
import math
import json
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.train import (
    ensure_dir,
    available_device,
    set_seed,
    TinyDecoderLM,
    SyntheticLMDataset,
    eval_perplexity_lm,
)


def evaluate_exp1(models_dir: str, image_dir: str, cfg: Dict) -> Dict:
    device = available_device()
    ensure_dir(image_dir)
    set_seed(int(cfg.get('seed', 42)))

    proto_ckpt = os.path.join(models_dir, 'exp1_protohop.pt')
    dense_ckpt = os.path.join(models_dir, 'exp1_dense.pt')
    assert os.path.isfile(proto_ckpt) and os.path.isfile(dense_ckpt), "Missing checkpoints for exp1. Run train first."

    proto_data = torch.load(proto_ckpt, map_location=device)
    dense_data = torch.load(dense_ckpt, map_location=device)

    pconf = proto_data['config']
    dconf = dense_data['config']

    model_proto = TinyDecoderLM(vocab_size=pconf['vocab_size'], d_model=pconf['d_model'], n_layers=pconf['n_layers'], n_heads=pconf['n_heads'], ffn_hidden=pconf['ffn_hidden'], proto=True, S=pconf['S'], k_top=pconf['k_top']).to(device)
    model_dense = TinyDecoderLM(vocab_size=dconf['vocab_size'], d_model=dconf['d_model'], n_layers=dconf['n_layers'], n_heads=dconf['n_heads'], ffn_hidden=dconf['ffn_hidden'], proto=False).to(device)
    model_proto.load_state_dict(proto_data['state_dict'])
    model_dense.load_state_dict(dense_data['state_dict'])

    vocab_size = int(cfg.get('vocab_size', 256))
    seq_len = int(cfg.get('seq_len', 256))
    batch_size = int(cfg.get('batch_size', 8))
    size = int(cfg.get('eval_size', 64))

    ds_bigram = SyntheticLMDataset(vocab_size=vocab_size, seq_len=seq_len, size=size, pattern='bigram')
    ds_needle = SyntheticLMDataset(vocab_size=vocab_size, seq_len=seq_len, size=size, pattern='needle')
    ds_kvcopy = SyntheticLMDataset(vocab_size=vocab_size, seq_len=seq_len, size=size, pattern='kvcopy')

    loader_bigram = DataLoader(ds_bigram, batch_size=batch_size, shuffle=False)
    loader_needle = DataLoader(ds_needle, batch_size=batch_size, shuffle=False)
    loader_kvcopy = DataLoader(ds_kvcopy, batch_size=batch_size, shuffle=False)

    ppl_proto_bigram = eval_perplexity_lm(model_proto, loader_bigram, device)
    ppl_dense_bigram = eval_perplexity_lm(model_dense, loader_bigram, device)
    ppl_proto_needle = eval_perplexity_lm(model_proto, loader_needle, device)
    ppl_dense_needle = eval_perplexity_lm(model_dense, loader_needle, device)
    ppl_proto_kvcopy = eval_perplexity_lm(model_proto, loader_kvcopy, device)
    ppl_dense_kvcopy = eval_perplexity_lm(model_dense, loader_kvcopy, device)

    print(f"[Eval Exp1] Perplexity (bigram): ProtoHop={ppl_proto_bigram:.2f} vs Dense={ppl_dense_bigram:.2f}")
    print(f"[Eval Exp1] Perplexity (needle):  ProtoHop={ppl_proto_needle:.2f} vs Dense={ppl_dense_needle:.2f}")
    print(f"[Eval Exp1] Perplexity (kvcopy):  ProtoHop={ppl_proto_kvcopy:.2f} vs Dense={ppl_dense_kvcopy:.2f}")

    return {
        'ppl': {
            'bigram': (ppl_proto_bigram, ppl_dense_bigram),
            'needle': (ppl_proto_needle, ppl_dense_needle),
            'kvcopy': (ppl_proto_kvcopy, ppl_dense_kvcopy),
        }
    }


def evaluate_all(models_dir: str, image_dir: str, cfg: Dict) -> Dict:
    results = {}
    if cfg.get('run_exp1', True):
        results['exp1'] = evaluate_exp1(models_dir, image_dir, cfg.get('exp1', {}))
    # Exp2 and Exp3 use synthetic in-training diagnostics; for brevity we evaluate exp1 thoroughly here.
    # You can extend this to reload exp2/exp3 checkpoints similarly if needed.
    return results
