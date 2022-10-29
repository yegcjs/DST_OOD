import pdb
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import json
from collections import namedtuple
from utils.DSTDataManager import DSTDataManager, parallel_tokenize
import torch
from torch.nn.utils.rnn import pad_sequence

class TrippyDataset(Dataset):
    def __init__(self, path, tokenizer, config, device, debug):
        super(TrippyDataset, self).__init__()
        with open(path, "r", encoding='utf-8') as f:
            self.raw_data = json.load(f)[:2048 if debug else None]
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.text_ids, self.text_mask, self.token_type_ids = self.init_text()
        self.state = self.init_state()
        self.inform = self.init_inform()
        self.gate, self.span_start, self.span_end, self.ref_slot = self.init_labels()
        self.states = [entry['state'] for entry in self.raw_data]

    def __getitem__(self, item):
        """
        :return: ref_slot, state, inform are sparse matrices
        """
        # ref slot value is + 1
        return self.text_ids[item], self.text_mask[item], self.token_type_ids[item], self.state[item].to_dense(), \
               self.inform[item].to_dense(), self.gate[item], self.span_start[item], self.span_end[item], \
               self.ref_slot[item].to_dense()-1, self.states[item]

    def __len__(self):
        return len(self.raw_data)

    def init_text(self):
        texts = [
            [entry["user_utterance"], f'{entry["system_utterance"]}[SEP]{entry["history"]}']
            for entry in self.raw_data
        ]   # follow official implementation of trippy
        text_ids, mask, token_type_ids = parallel_tokenize(self.tokenizer, texts, trippy_tokenizer_collate)
        return text_ids.to(self.device), mask.to(self.device), token_type_ids.to(self.device)

    def init_state(self):
        indices = [[i, j] for i, entry in enumerate(self.raw_data) for j, slot in enumerate(entry['known_slots'])]
        indices, values = torch.tensor(indices).T, [1] * len(indices)
        return torch.sparse_coo_tensor(indices, values, (len(self), len(self.config.idx2slot)), device=self.device)

    def init_inform(self):
        indices = [[i, j] for i, entry in enumerate(self.raw_data) for j, slot in enumerate(entry['informed_slots'])]
        indices, values = torch.tensor(indices).T, [1] * len(indices)
        print(indices.shape)
        return torch.sparse_coo_tensor(indices, values, (len(self), len(self.config.idx2slot)), device=self.device)

    def init_labels(self):
        num_slots = len(self.config.idx2slot)
        label2idx, slot2idx = self.config.label2idx, self.config.slot2idx
        labels_shape = (len(self), num_slots)

        gates = torch.zeros(labels_shape, dtype=torch.long, device=self.device)
        span_start = torch.zeros(labels_shape, dtype=torch.long, device=self.device) - 1  # -1 as ignore index
        span_end = torch.zeros(labels_shape, dtype=torch.long, device=self.device) - 1  # -1 as ignore index
        ref_slot_indices, ref_slot_values = [], []
        for i, entry in enumerate(self.raw_data):
            user_utt_end = self.locate_user_utterance_range(self.text_ids[i])
            for slot, (label, x) in entry['labels'].items():  # x can be 'none' | <reference_slot> | <span_value>
                j = slot2idx[slot]
                gates[i, j] = label2idx[label]
                if label == "copy_value":
                    span_start[i, j], span_end[i, j] = self.get_span(self.text_ids[i, :user_utt_end], x)
                elif label == "refer":
                    ref_slot_indices.append((i, j))
                    ref_slot_values.append(slot2idx[x])
        if len(ref_slot_indices) == 0:
            ref_slot_indices.append((0, 0))
            ref_slot_values.append(0)
            print("No ref indices !!!")
        ref_slot_indices = torch.tensor(ref_slot_indices).T
        refer_slot = torch.sparse_coo_tensor(
            ref_slot_indices, [v + 1 for v in ref_slot_values], labels_shape, device=self.device
        )  # v+1 because we expect -1 as fill value
        return gates, span_start, span_end, refer_slot

    def locate_user_utterance_range(self, input_ids):
        for i, token in enumerate(input_ids):
            if token.item() == self.tokenizer.sep_token_id:
                return i  # return the position of the first [SEP]

    def get_span(self, text_ids: torch.Tensor, span_value: str) -> (int, int):
        text_ids = text_ids.tolist()
        span_ids = self.tokenizer.encode(span_value)[1:-1]
        span_token_length = len(span_ids)
        for p in range(len(text_ids), 0, -1):  # get the rightmost one
            if text_ids[p - span_token_length:p] == span_ids:
                return p - span_token_length, p - 1
        print("span error:", text_ids, span_ids, span_value)

    # @staticmethod
    # def collate_fn(batch):
    #     """
    #     :param batch: List[ (text_ids[item], text_mask[item], token_type_ids[item], state[item], inform[item], \
    #                          gates[item], span_start[item], span_end[item], ref_slot[item]) ]
    #     :return: 9 matrices
    #     """
    #     ret = [
    #         torch.stack([entry[i] for entry in batch], dim=0) for i in range(len(batch[0])-1)
    #     ] + [[entry[-1] for entry in batch]]  # entry[-1] == state
    #     return tuple(ret)


trippy_config_t = namedtuple("trippy_config_t", ['idx2label', 'label2idx', 'idx2slot', 'slot2idx', 'value_maps'])


class TrippyDataManager(DSTDataManager):
    def __init__(self, tokenizer, config_file, debug=False):
        super(TrippyDataManager, self).__init__(tokenizer=tokenizer, debug=debug)
        with open(config_file, "r", encoding='utf-8') as f:
            config = json.load(f)
        idx2label = [label.lower() for label in config['class_types']]
        label2idx = {label: idx for idx, label in enumerate(idx2label)}
        idx2slot = [slot.lower() for slot in config['slots']]
        slot2idx = {slot: idx for idx, slot in enumerate(idx2slot)}
        value_maps = config['label_maps']
        self.config = trippy_config_t(idx2label, label2idx, idx2slot, slot2idx, value_maps)


    @staticmethod
    def collate_fn(batch):
        ret = [
            torch.stack([entry[i] for entry in batch], dim=0) for i in range(len(batch[0]) - 1)
        ] + [[entry[-1] for entry in batch]]  # entry[-1] == state
        return tuple(ret)

    def load_dataset(self, data_split, path, device):
        self.datasets[data_split] = TrippyDataset(path, self.tokenizer, self.config, device, self.debug)


def trippy_tokenizer_collate(tokenized_texts):
    sep_token_id = 102
    input_ids = pad_sequence([
        torch.tensor(seq_ids) if len(seq_ids) < 512 else torch.tensor(seq_ids[:511]+[sep_token_id])
        for batch in tokenized_texts for seq_ids in batch['input_ids']
        ], batch_first=True)
    masks = pad_sequence([
        torch.tensor(seq_mask) if len(seq_mask) < 512 else torch.tensor(seq_mask[:512])
        for batch in tokenized_texts for seq_mask in batch['attention_mask']
        ], batch_first=True)
    token_type_ids = pad_sequence([
        torch.tensor(type_ids) if len(type_ids) <= 512 else torch.tensor(type_ids[:512])
        for batch in tokenized_texts for type_ids in batch['token_type_ids']
        ], batch_first=True)
    return input_ids, masks, token_type_ids
