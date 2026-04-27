import argparse
import os
import random

import numpy as np
import torch

from electro_ts.exp.aemwe import Exp_Long_Term_Forecast


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ElectroTS')

    # Core
    parser.add_argument('--task', type=str, default='train', choices=['train', 'test', 'predict'])
    parser.add_argument('--model_id', type=str, required=True, help='Experiment identifier')
    parser.add_argument('--model', type=str, default='ElectroTS', choices=['ElectroTS'])

    # Data
    parser.add_argument('--data', type=str, default='AEMWE')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Dataset root directory (must contain an "all/" subdirectory)')
    parser.add_argument('--time_cutoff', type=float, default=500.0,
                        help='Minimum operating duration (h) for a cell to qualify as test data')
    parser.add_argument('--restart_path', type=str, default=None,
                        help='Checkpoint path for warm-starting the model')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # Sequence lengths
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='Decoder start-token overlap length')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction length')

    # Model architecture
    parser.add_argument('--enc_in', type=int, default=1, help='Encoder input channels')
    parser.add_argument('--dec_in', type=int, default=1, help='Decoder input channels')
    parser.add_argument('--c_out', type=int, default=1, help='Output channels')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--d_core', type=int, default=512, help='STAR core dimension')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'gelu'])
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--use_norm', type=int, default=1,
                        help='Enable reversible instance normalization (1=on, 0=off)')

    # Training
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lradj', type=str, default='type1',
                        choices=['type1', 'type2', 'cosine', 'constant'])
    parser.add_argument('--use_amp', action='store_true', help='Automatic mixed precision')
    parser.add_argument('--save_model', action='store_true', help='Keep checkpoints after training')

    # Hardware
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    return parser


def _run_one(args, test_txt_path: str) -> dict:
    setting = f'{args.model_id}/idx{args.test_idx:02d}/'
    exp = Exp_Long_Term_Forecast(args)
    exp.train(setting)
    exp.test(setting)
    torch.cuda.empty_cache()
    result = exp.predict(setting, test_txt_path)
    torch.cuda.empty_cache()
    return result


def main():
    # Reproducibility
    random.seed(2)
    torch.manual_seed(2)
    np.random.seed(2)
    torch.set_num_threads(6)

    args = _build_parser().parse_args()
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        device_ids = args.devices.replace(' ', '').split(',')
        args.device_ids = [int(d) for d in device_ids]
        args.gpu = args.device_ids[0]

    print('Args:', args)

    # Discover test cells via leave-one-out split
    all_dir = os.path.join(args.root_path, 'all')
    txt_files = sorted(f for f in os.listdir(all_dir) if f.endswith('.txt'))
    arrays = [np.loadtxt(os.path.join(all_dir, f)) for f in txt_files]
    test_files = [f for f, arr in zip(txt_files, arrays) if arr[:, 0].max() > args.time_cutoff]
    print(f'Test cell count: {len(test_files)}')

    # Leave-one-out training loop
    result_list = []
    for j, fname in enumerate(test_files):
        print(f'\n===== Test cell {j} / {len(test_files) - 1}: {fname} =====')
        args.test_idx = j
        result_list.append(_run_one(args, os.path.join(all_dir, fname)))

    # Aggregate and print results
    agg: dict[str, list] = {k: [] for k in result_list[0]}
    for res in result_list:
        for k, v in res.items():
            agg[k].append(v)

    print('\n===== Aggregated Results =====')
    for k, values in agg.items():
        valid = [x for x in values if x is not None]
        print(f'{k}: {values}')
        if valid:
            arr = np.array(valid)
            print(f'  -> mean={arr.mean():.4f}  std={arr.std():.4f}')
        else:
            print(f'  -> mean=None  std=None')
