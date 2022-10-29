import json
import os.path

from DataPreprocessor import DataPreprocessor, CommonUtils


class SimpleTODDataPreprocessor(DataPreprocessor):
    def __init__(self, data_file):
        super(SimpleTODDataPreprocessor, self).__init__(data_file)

    def create_dataset(self, filter_file, target_file):
        raw_dataset = self.filter(filter_file)
        print(filter_file, len(raw_dataset))
        prefix = "<|endoftext|> <|context|>"
        middle = "<|endofcontext|> <|belief|>"
        suffix = "<|endofbelief|> <|endoftext|>"
        simpletod_data = []
        for dialog_id, dialog_content in raw_dataset.items():
            history = ""
            for i, turn in enumerate(dialog_content['log']):
                speaker = "user" if i % 2 == 0 else "system"

                # invalid data entry: user, system were not taking turns
                if speaker == "user" and turn['metadata'] != {}:
                    break
                if speaker == "system" and turn['metadata'] == {}:
                    break

                if speaker == "system":
                    believes = self.state_to_belief_str(CommonUtils.metadata_to_state(turn['metadata']))
                    item = " ".join(f"{prefix} {history} {middle} {believes} {suffix}".split())
                    simpletod_data.append(item)
                utterance = turn['text'].lower().replace('\n', ' ').replace(',', ' , ').replace(' . ', ' . ')
                utterance = " ".join(utterance.split())
                history += f"<|{speaker}|> {utterance} "

        with open(target_file, "w", encoding='utf-8') as f:
            json.dump(simpletod_data, f, indent=2)

    @staticmethod
    def state_to_belief_str(state):
        beliefs = []
        for key, value in state.items():
            if value != '':
                beliefs.append(f"{key.replace('-', ' ').replace('_', ' ')} {value}")
        return " , ".join(list(beliefs))


def main():
    data_processor = SimpleTODDataPreprocessor("MultiWOZ2_1/data.json")
    if not os.path.exists("MultiWOZ2_1_SimpleTOD"):
        os.mkdir("MultiWOZ2_1_SimpleTOD")
    data_processor.create_dataset("MultiWOZ2_1/trainListFile.txt", "MultiWOZ2_1_SimpleTOD/train.json")
    data_processor.create_dataset("MultiWOZ2_1/valListFile.txt", "MultiWOZ2_1_SimpleTOD/valid.json")
    data_processor.create_dataset("MultiWOZ2_1/testListFile.txt", "MultiWOZ2_1_SimpleTOD/test.json")


if __name__ == '__main__':
    main()
