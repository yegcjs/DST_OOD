import PrepareOoD
import os
from SGDBaseline_DataPreprocessor import SGDBaselineDataPreprocessor
from SimpleTOD_DataPreprocessor import SimpleTODDataPreprocessor
from Trippy_DataPreprocessor import TrippyDataPreprocessor
from Trade_DataPreprocessor import TradeDataPreprocessor

TEST_DATA_TYPES = []


def init_simpletod(src_base, target_base):
    SimpleTODDataPreprocessor(f"{src_base}/train/train.json").create_dataset(None, f"{target_base}/train.json")
    SimpleTODDataPreprocessor(f"{src_base}/train/valid.json").create_dataset(None, f"{target_base}/valid.json")
    # for pref in ['test', 'generated']:
    for data_type in TEST_DATA_TYPES:
        SimpleTODDataPreprocessor(
            f"{src_base}/{data_type}/data.json"
        ).create_dataset(None, f"{target_base}/{data_type}.json")


def init_trippy(src_base, target_base):
    config = "auxfiles/multiwoz21config.json"
    TrippyDataPreprocessor(
        config, f"{src_base}/train/train_dialog_acts.json", f"{src_base}/train/train.json"
    ).create_dataset(None, f"{target_base}/train.json")
    TrippyDataPreprocessor(
        config, f"{src_base}/train/valid_dialog_acts.json", f"{src_base}/train/valid.json"
    ).create_dataset(None, f"{target_base}/valid.json")
    # for pref in ['test', 'generated']:
    for data_type in TEST_DATA_TYPES:
        TrippyDataPreprocessor(
            config, f"{src_base}/{data_type}/system_acts.json", f"{src_base}/{data_type}/data.json"
        ).create_dataset(None, f"{target_base}/{data_type}.json")


def init_trade(src_base, target_base):
    TradeDataPreprocessor(
        f"{src_base}/train/train.json", f"{src_base}/train/train_dialog_acts.json"
    ).create_dataset(None, f"{target_base}/train.json")
    TradeDataPreprocessor(
        f"{src_base}/train/valid.json", f"{src_base}/train/valid_dialog_acts.json"
    ).create_dataset(None, f"{target_base}/valid.json")
    # for pref in ['test', 'generated']:
    for data_type in TEST_DATA_TYPES:
        TradeDataPreprocessor(
            f"{src_base}/{data_type}/data.json", f"{src_base}/{data_type}/system_acts.json"
        ).create_dataset(None, f"{target_base}/{data_type}.json")


if __name__ == '__main__':
    PrepareOoD.PrepareOoD(
        full_data_path='MultiWOZ2_3/data.json',
        full_acts_path='MultiWOZ2_3/dialogue_acts.json',
        train_list_file='auxfiles/trainListFile.txt',
        valid_list_file='auxfiles/valListFile.txt',
        test_list_file='auxfiles/testListFile.txt',
        seed=42
    ).run("MultiWOZ_OoD")
    TEST_DATA_TYPES = os.listdir("MultiWOZ_OoD")
    TEST_DATA_TYPES.remove("train")
    init_method = {
        'SimpleTOD': init_simpletod,
        'Trippy': init_trippy,
        'Trade': init_trade
    }
    for method in ['SimpleTOD', 'Trippy', 'Trade']: 
        target_dir = f"MultiWOZ_OoD_{method}"
        os.makedirs(target_dir, exist_ok=True)
        init_method[method]("MultiWOZ_OoD", target_dir)
