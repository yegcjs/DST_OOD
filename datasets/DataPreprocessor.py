import abc
import json
from abc import ABCMeta


class DataPreprocessor(metaclass=ABCMeta):
    def __init__(self, data_path):
        with open(data_path, "r", encoding='utf-8') as f:
            self.full_data = json.load(f)

    def filter(self, dialog_list_file):
        if dialog_list_file is None:
            return self.full_data   # if no need to filter
        # pick a subset of all dialogs, e.g. training set, validation set, or test set
        with open(dialog_list_file, "r", encoding='utf-8') as f:
            dialog_subset = {line.strip() for line in f}
        return {
            dialog_id: dialog_content for dialog_id, dialog_content in self.full_data.items()
            if dialog_id in dialog_subset
        }

    @abc.abstractmethod
    def create_dataset(self, filter_file, target_file):
        pass


class CommonUtils:
    @staticmethod
    def metadata_to_state(metadata):
        state = {}
        for domain, metaslots in metadata.items():
            booked = {}
            for slot, value in metaslots['book'].items():
                slot = slot.lower()
                if slot == 'booked':
                    if len(value) == 0:
                        continue
                    for booked_slot, booked_value in value[0].items():
                        booked_slot = booked_slot.lower()
                        booked[f"{domain}-{booked_slot}"] = booked_value
                else:
                    state[f"{domain}-book_{slot}"] = value
            for slot, value in metaslots['semi'].items():
                slot = slot.lower()
                if f"{domain}-{slot}" in booked:
                    state[f"{domain}-{slot}"] = booked[f"{domain}-{slot}"]
                else:
                    state[f"{domain}-{slot}"] = value
        return state

