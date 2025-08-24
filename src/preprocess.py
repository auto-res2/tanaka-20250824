import os
import json
from typing import Dict

from src.train import ensure_dir, set_seed


def preprocess(cfg: Dict) -> Dict:
    # Synthetic data: ensure directories and write a small manifest for reproducibility.
    data_dir = cfg.get('data_dir', 'data')
    ensure_dir(data_dir)
    ensure_dir(cfg.get('models_dir', 'models'))
    ensure_dir(cfg.get('image_dir', '.research/iteration1/images'))

    set_seed(int(cfg.get('seed', 42)))

    manifest = {
        'note': 'Synthetic datasets are generated on-the-fly at train/eval time.',
        'exp1': cfg.get('exp1', {}),
        'exp2': cfg.get('exp2', {}),
        'exp3': cfg.get('exp3', {}),
    }
    with open(os.path.join(data_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Preprocess complete. Wrote manifest to {os.path.join(data_dir, 'manifest.json')}.")
    return manifest
