import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    # shared
    parser.add_argument("--method", required=True)  # simpletod|trippy|...
    parser.add_argument("--devices", nargs='+', type=int)
    parser.add_argument("--log", type=str, required=True, help="directory to save logs")
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--save_memory", action='store_true', default=False,
                        help="whether to enable gradient checkpoint to save memory used")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="only a small subset of data is loaded if debug is enabled")
    parser.add_argument("--pretrained", type=str, required='Trade' not in sys.argv,
                        help="pretrained model(bert, gpt2, etc.)'s directory")
    # training
    training_flag = '--train' in sys.argv
    parser.add_argument("--train", type=str, help="directory to training set and validation set")
    parser.add_argument("--saveto", type=str, required=training_flag,
                        help="directory to save model")
    parser.add_argument("--epochs", type=int, required=training_flag,
                        help="epochs to train")
    parser.add_argument("--early_stop", type=int, required=training_flag,
                        help="# of rounds to train after achieving best validation metrics")
    parser.add_argument("--update_steps", type=int, required=training_flag,
                        help="# of steps to accumulate gradients, update_steps * batch_size * num_devices = real "
                             "batch_size")
    parser.add_argument("--eval_steps", type=int, required=training_flag,
                        help="# of steps between two evaluations")
    parser.add_argument("--first_eval", type=int, required=training_flag,
                        help="times of evaluations to skip at the beginning")
    parser.add_argument("--seed", type=int, required=training_flag, default=42,
                        help="random seed")
    parser.add_argument("--port", type=str, required=training_flag, default='30000',
                        help="port for distributed training communication")

    # testing
    testing_flag = '--test' in sys.argv
    parser.add_argument("--test", type=str, help="test set file")
    parser.add_argument("--ood", required=testing_flag, action='store_true', default=False, help="is doing ood test?")
    parser.add_argument("--checkpoint", type=str, required=testing_flag, help="directory to trained model")
    parser.add_argument("--dump_path", type=str, help="path to dump test info")

    # trippy
    trippy_flag = 'trippy' in sys.argv
    parser.add_argument("--config", type=str, required=trippy_flag,
                        help="config file for dataset")
    parser.add_argument("--lambda_gate", type=float, required=trippy_flag and training_flag,
                        help="weight for gate loss")
    parser.add_argument("--lambda_span", type=float, required=trippy_flag and training_flag,
                        help="weight for span loss")
    parser.add_argument("--lambda_refer", type=float, required=trippy_flag and training_flag,
                        help="weight for ref loss")

    # sgdbaseline
    sgdbaseline_flag = 'sgdbaseline' in sys.argv
    parser.add_argument("--schema", type=str, required=sgdbaseline_flag,
                        help="path of schema.json file")

    # trade
    if 'trade' in sys.argv:
        # Training Setting
        parser.add_argument('-ds', '--dataset', help='dataset', required=False, default="multiwoz")
        parser.add_argument('-t', '--task', help='Task Number', required=False, default="dst")
        parser.add_argument('-path', '--path', help='path of the file to load', required=False)
        parser.add_argument('-sample', '--sample', help='Number of Samples', required=False, default=None)
        parser.add_argument('-patience', '--patience', help='', required=False, default=6, type=int)
        parser.add_argument('-es', '--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False,
                            default='BLEU')
        parser.add_argument('-all_vocab', '--all_vocab', help='', required=False, default=1, type=int)
        parser.add_argument('-imbsamp', '--imbalance_sampler', help='', required=False, default=0, type=int)
        parser.add_argument('-data_ratio', '--data_ratio', help='', required=False, default=100, type=int)
        parser.add_argument('-um', '--unk_mask', help='mask out input token to UNK', type=int, required=False,
                            default=1)
        parser.add_argument('-bsz', '--batch', help='Batch_size', required=False, type=int)

        # Testing Setting
        parser.add_argument('-rundev', '--run_dev_testing', help='', required=False, default=0, type=int)
        parser.add_argument('-viz', '--vizualization', help='vizualization', type=int, required=False, default=0)
        parser.add_argument('-gs', '--genSample', help='Generate Sample', type=int, required=False, default=0)
        parser.add_argument('-evalp', '--evalp', help='evaluation period', required=False, default=1)
        parser.add_argument('-an', '--addName', help='An add name for the save folder', required=False, default='')
        parser.add_argument('-eb', '--eval_batch', help='Evaluation Batch_size', required=False, type=int, default=0)

        # Model architecture
        parser.add_argument('-gate', '--use_gate', help='', required=False, default=1, type=int)
        parser.add_argument('-le', '--load_embedding', help='', required=False, default=0, type=int)
        parser.add_argument('-femb', '--fix_embedding', help='', required=False, default=0, type=int)
        parser.add_argument('-paral', '--parallel_decode', help='', required=False, default=0, type=int)

        # Model Hyper-Parameters
        parser.add_argument('-dec', '--decoder', help='decoder model', required=False)
        parser.add_argument('-hdd', '--hidden', help='Hidden size', required=False, type=int, default=400)
        parser.add_argument('-lr', '--learn', help='Learning Rate', required=False, type=float)
        parser.add_argument('-dr', '--drop', help='Drop Out', required=False, type=float)
        parser.add_argument('-lm', '--limit', help='Word Limit', required=False, default=-10000)
        parser.add_argument('-clip', '--clip', help='gradient clipping', required=False, default=10, type=int)
        parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False,
                            default=0.5)
        # parser.add_argument('-l','--layer', help='Layer Number', required=False)

        # Unseen Domain Setting
        parser.add_argument('-l_ewc', '--lambda_ewc', help='regularization term for EWC loss', type=float,
                            required=False,
                            default=0.01)
        parser.add_argument('-fisher_sample', '--fisher_sample', help='number of sample used to approximate fisher mat',
                            type=int, required=False, default=0)
        parser.add_argument("--all_model", action="store_true")
        parser.add_argument("--domain_as_task", action="store_true")
        parser.add_argument('--run_except_4d', help='', required=False, default=1, type=int)
        parser.add_argument("--strict_domain", action="store_true")
        parser.add_argument('-exceptd', '--except_domain', help='', required=False, default="", type=str)
        parser.add_argument('-onlyd', '--only_domain', help='', required=False, default="", type=str)

        args = vars(parser.parse_args())
        if args["load_embedding"]:
            args["hidden"] = 400
            print("[Warning] Using hidden size = 400 for pretrained word embedding (300 + 100)...")
        if args["fix_embedding"]:
            args["addName"] += "FixEmb"
        if args["except_domain"] != "":
            args["addName"] += "Except" + args["except_domain"]
        if args["only_domain"] != "":
            args["addName"] += "Only" + args["only_domain"]

    return parser.parse_args()
