import argparse
import os
import json
import yaml

from src.preprocess import preprocess
from src.train import train_all_experiments
from src.evaluate import evaluate_all


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='ProtoHop TinyLM — experiments runner')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to YAML config')
    parser.add_argument('--only-preprocess', action='store_true', help='Only run preprocessing')
    parser.add_argument('--only-train', action='store_true', help='Only run training')
    parser.add_argument('--only-eval', action='store_true', help='Only run evaluation')
    parser.add_argument('--quick', action='store_true', help='Override config with a quick test setup')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.quick:
        # Small, fast settings to validate end-to-end execution on modest GPUs/CPUs
        cfg['exp1'].update({'train_size': 32, 'seq_len': 128, 'batch_size': 8, 'latency_contexts': [128, 256], 'latency_steps': 2})
        cfg['exp2'].update({'train_size': 32, 'src_len': 512, 'tgt_len': 32, 'chunk_size': 64, 'batch_size': 4})
        cfg['exp3'].update({'train_size': 128, 'batch_size': 32})

    # Ensure directories
    os.makedirs(cfg.get('data_dir', 'data'), exist_ok=True)
    os.makedirs(cfg.get('models_dir', 'models'), exist_ok=True)
    os.makedirs(cfg.get('image_dir', '.research/iteration1/images'), exist_ok=True)

    if args.only_preprocess:
        preprocess(cfg)
        return

    if args.only_train:
        preprocess(cfg)
        results = train_all_experiments(cfg)
        print(json.dumps(results, indent=2))
        return

    if args.only_eval:
        results = evaluate_all(cfg.get('models_dir', 'models'), cfg.get('image_dir', '.research/iteration1/images'), cfg)
        print(json.dumps(results, indent=2))
        return

    # Default: run full pipeline
    preprocess(cfg)
    results = train_all_experiments(cfg)
    eval_results = evaluate_all(cfg.get('models_dir', 'models'), cfg.get('image_dir', '.research/iteration1/images'), cfg)

    print("\n==== Train Results ====")
    print(json.dumps(results, indent=2))
    print("\n==== Eval Results ====")
    print(json.dumps(eval_results, indent=2))


if __name__ == '__main__':
    main()
