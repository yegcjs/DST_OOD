import argparse
from subprocess import Popen
import time
from copy import deepcopy
import sys
import os
import pynvml


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str)
    parser.add_argument("--devices", nargs='+', type=int)
    parser.add_argument("--debug", action='store_true', default=False)

    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--port", type=str, default="30000")

    parser.add_argument("--ood", action='store_true', default=False)

    parser.add_argument("--pretrained", type=str, required=True)
    # following are for single test
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--recover_from", type=str)

    return parser.parse_args()


def gpu_ok(gpu_ids):
    gpus = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_ids]
    for gpu in gpus:
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu)
        if meminfo.free < 9 * 1024 * 1024 * 1024:
            return False
    return True


def run_method(args):
    method = args.method
    METHOD_DICT = {
        'SimpleTOD': {
            'batch_size': 1,
            'pretrained': args.pretrained,
            'first_eval': 48
        },
        'Trippy': {
            'batch_size': 32,
            'pretrained': args.pretrained,
            'first_eval': 5
        },
        'Trade': {
            'batch_size': 32,
            'pretrained': '_',
            'first_eval': 5
        }
    }
    general_command = [
        'python', 'main.py',
        '--method', method.lower(),
        '--log', f'{method}/logs',
        '--batch_size', str(METHOD_DICT[method]['batch_size']),
    ]

    if method == 'Trippy':
        general_command += [
            "--save_memory",
            "--config", "datasets/auxfiles/multiwoz21config.json",
            "--lambda_gate", "0.8",
            "--lambda_span", "0.1",
            "--lambda_refer", "0.1"
        ]
    if args.debug:
        general_command.append("--debug")

    time_stamp = time.strftime("%b%d_%H%M%S", time.localtime())

    full_epoch_size = 2048 if args.debug else 2 ** 16
    data_path = f"datasets/MultiWOZ2_1_{method}" if not args.ood else f"datasets/MultiWOZ_OoD_{method}/"
    # train
    if args.train:
        train_command = deepcopy(general_command)
        train_command += [
            "--train", data_path,
            "--pretrained", METHOD_DICT[method]['pretrained'],
            "--saveto", f'{method}/checkpoints/{time_stamp}',
            "--epoch", '100000',
            '--early_stop', '3',
            '--update_steps', str(128 // len(args.devices) // METHOD_DICT[method]['batch_size']),
            '--eval_steps', str(full_epoch_size // len(args.devices) // METHOD_DICT[method]['batch_size']),
            '--first_eval', str(METHOD_DICT[method]['first_eval']),
            '--seed', '42',
            '--port', args.port,
            '--devices'
        ]
        for device in args.devices:
            train_command.append(str(device))
        if args.recover_from is not None:
            train_command += ['--checkpoint', args.recover_from]
        print(train_command)
        while not gpu_ok(args.devices):
            pass
        proc = Popen(train_command)
        proc.wait()

    if args.checkpoint is None:
        args.checkpoint = f'{method}/checkpoints/{time_stamp}'
    if args.test:
        test_command = deepcopy(general_command)
        test_command += [
            '--test', f'datasets/MultiWOZ2_1_{method}/test.json',
            '--pretrained', METHOD_DICT[method]['pretrained'],
            '--checkpoint', args.checkpoint,
            '--devices', str(args.devices[0])
        ]
        while not gpu_ok(args.devices):
            pass
        proc = Popen(test_command)
        proc.wait()

    if args.ood:
        for pref in ['generated', 'test']:
            for data_type in [
                    "id_dialogs", "id_hist_contextual_ood_utt",
                    "id_hist_non_contextual_ood_utt",
                    "ood_hist_contextual_ood_utt",
                    "ood_hist_non_contextual_ood_utt"
                ]:
                test_command = deepcopy(general_command)
                test_command += [
                    '--test', f"{data_path}/{pref}_{data_type}.json",
                    '--pretrained', METHOD_DICT[method]['pretrained'],
                    '--checkpoint', args.checkpoint,
                    '--devices', str(args.devices[0]),
                    '--dump_path', f"{method}/dump/{pref}_{data_type}.json",
                    '--ood'
                ]
                proc = Popen(test_command)
                proc.wait()



def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    pynvml.nvmlInit()
    args = parse_arguments()
    print(args)
    run_method(args)


if __name__ == '__main__':
    main()
