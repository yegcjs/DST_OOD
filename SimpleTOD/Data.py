import json
import abc
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from utils.DSTDataManager import DSTDataManager
import torch


class SimpleTODDataset(Dataset):
    def __init__(self, path, device_id, tokenizer, debug=False):
        super(SimpleTODDataset, self).__init__()
        with open(path, "r", encoding='utf-8') as f:
            full_lines = [line.strip() for line in json.load(f)[:1024 if debug else None]]

        full_data = tokenizer(full_lines)
        self.full_ids = [
            torch.tensor(item, device=f"cuda:{device_id}") for item in full_data['input_ids'] if len(item) <= 1024
        ]
        self.full_masks = [
            torch.tensor(item, device=f"cuda:{device_id}") for item in full_data['attention_mask'] if len(item) <= 1024
        ]

        self.dialog_states = [
            {state.strip() for state in line.split('<|belief|>')[-1].split('<|endofbelief|>')[0].split(',')
             if state.strip() != '' and 'none' not in state.strip() and 'not mentioned' not in state.strip() }  #   FIXME
            for line, tokens in zip(full_lines, full_data['input_ids']) if len(tokens) <= 1024
        ]

        dialog_history = [
            line.split('<|belief|>')[0] + '<|belief|>'
            for line, tokens in zip(full_lines, full_data['input_ids']) if len(tokens) <= 1024
        ]
        self.history = [torch.tensor(item, device=f"cuda:{device_id}") for item in
                        tokenizer(dialog_history)['input_ids']]

    def __getitem__(self, item):
        return self.full_ids[item], self.full_masks[item], self.history[item], self.dialog_states[item]

    def __len__(self):
        return len(self.full_ids)

    # @staticmethod
    # def collate_fn(batch):
    #     """
    #     :param batch: List[Tuple[full_ids, full_mask, history_mask]]
    #     :return: Tuple[full_ids, full_masks, history, state]
    #     """
    #     full_ids = pad_sequence([entry[0] for entry in batch], batch_first=True)
    #     full_masks = pad_sequence([entry[1] for entry in batch], batch_first=True)
    #     histories = [entry[2] for entry in batch]
    #     states = [entry[3] for entry in batch]
    #     return full_ids, full_masks, histories, states


class SimpleTODDataManager(DSTDataManager):
    def __init__(self, tokenizer, debug=False):
        super(SimpleTODDataManager, self).__init__(tokenizer=tokenizer, debug=debug)

    @staticmethod
    def collate_fn(batch):
        """
        :param batch: List[Tuple[full_ids, full_mask, history_mask]]
        :return: Tuple[full_ids, full_masks, history, state]
        """
        full_ids = pad_sequence([entry[0] for entry in batch], batch_first=True)
        full_masks = pad_sequence([entry[1] for entry in batch], batch_first=True)
        histories = [entry[2] for entry in batch]
        states = [entry[3] for entry in batch]
        return full_ids, full_masks, histories, states

    def load_dataset(self, data_split, path, device):
        print(f"Loading {path} as {data_split}")
        self.datasets[data_split] = SimpleTODDataset(path, device, self.tokenizer, debug=self.debug)

