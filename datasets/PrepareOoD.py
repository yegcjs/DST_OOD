import copy
import json
import os.path
import random
from collections import defaultdict
from tqdm import tqdm

ACT_SLOT_TO_STATE_SLOT = {
    'stay': 'book_stay',
    'price': 'pricerange',
    'parking': 'parking',
    'stars': 'stars',
    'internet': 'internet',
    'arrive': 'arriveBy',
    'people': 'book_people',
    'time': 'book_time',
    'area': 'area',
    'food': 'food',
    'depart': 'departure',
    'type': 'type',
    'dest': 'destination',
    'day': 'book_day',  # train day, hotel book_day....
    'name': 'name',
    'leave': 'leaveAt'
}
STATE_SLOT_TO_ACT_SLOT = {
    'book stay': 'Stay',
    'pricerange': 'Price',
    'parking': 'Parking',
    'stars': 'Stars',
    'internet': 'Internet',
    'arriveby': 'Arrive',
    'book people': 'People',
    'book time': 'Time',
    'area': 'Area',
    'food': 'Food',
    'departure': 'Depart',
    'type': 'Type',
    'destination': 'Dest',
    'book day': 'Day',
    'day': 'Day',
    'name': 'Name',
    'leaveat': 'Leave'
}


class PrepareOoD:
    def __init__(self, full_data_path, full_acts_path, train_list_file, valid_list_file, test_list_file, seed=42):
        random.seed(seed)
        print("reading original datasets for initialization...")
        with open(full_data_path, "r", encoding='utf-8') as f:
            self.full_data = json.load(f)
        with open(full_acts_path, "r", encoding='utf-8') as f:
            self.full_sys_acts = json.load(f)
        self.ori_train_data = self.drop_unknown_domain(self.filter_dialog(train_list_file), {'hospital', 'police'})
        self.ori_valid_data = self.filter_dialog(valid_list_file)
        self.ori_test_data = self.filter_dialog(test_list_file)
        self.id_act_comb, \
        self.id_slot_values, \
        self.id_sys_user_map, \
        self.id_history_user_map = self.init_in_distribution()
        self.utterance_with_act = self.init_utterance_with_act(self.ori_test_data)
        self.ood_dialog_history = self.init_ood_dialog_history(self.ori_test_data)

        void_state = {}
        for _, dialog_content in self.ori_test_data.items():
            for turn in dialog_content['log']:
                for domain, metaslots in turn['metadata'].items():
                    if domain not in void_state:
                        void_state[domain] = {}
                    for metaslot, slot_values in metaslots.items():
                        if metaslot not in void_state[domain]:
                            void_state[domain][metaslot] = {}
                        for slot, _ in slot_values.items():
                            if slot == 'booked':
                                void_state[domain][metaslot][slot] = []
                            else:
                                void_state[domain][metaslot][slot] = ""
        self.void_state = void_state
        print("initialization done\n")

    def init_in_distribution(self):
        """
        data: training data
        returns: set[action combination]
                 set[values]
                 map[(system action, user action)]
                 map[((system action 0, user action 0, ..., system action t), user action t)]
        """
        id_actions = set()
        id_slot_values = set()
        id_sys_user_map = defaultdict(set)
        id_history_user_map = defaultdict(set)
        for dialog_id, dialog_content in self.ori_train_data.items():
            dialog_history = []
            system_action = frozenset()
            for turn_id, turn in enumerate(dialog_content['log']):
                current_action_wo_v = frozenset(self.extract_action(turn['dialog_act'], with_values=False))
                current_action_w_v = frozenset(self.extract_action(turn['dialog_act'], with_values=True))
                id_actions.add(current_action_wo_v)
                for _, _, _, value in current_action_w_v:
                    id_slot_values.add(value)
                if turn_id % 2 == 0:  # user
                    id_sys_user_map[system_action].add(current_action_wo_v)
                    id_history_user_map[tuple(dialog_history)].add(current_action_wo_v)
                else:
                    system_action = copy.deepcopy(current_action_wo_v)
                dialog_history.append(current_action_wo_v)
        return id_actions, id_slot_values, id_sys_user_map, id_history_user_map

    def init_utterance_with_act(self, data):
        """
        param data: dialogs
        returns: List[(utterance, action)]
        """
        utt_with_act = []
        for dialog_id, dialog_content in data.items():
            for turn_id, turn in enumerate(dialog_content['log']):
                if turn_id % 2 == 1:
                    continue
                utterance = turn['text']
                # action_w_val = frozenset(self.extract_action(turn['dialog_act'], with_values=True))
                # action_wo_val = frozenset(self.extract_action(turn['dialog_act'], with_values=False))
                utt_with_act.append((utterance, turn['dialog_act']))
        return utt_with_act

    def drop_unknown_domain(self, data, unk_domains):
        unk_domains = set(unk_domains)
        return {
            dialog_id: dialog_content for dialog_id, dialog_content in data.items()
            if len(set(dialog_content['new_goal'].keys()) & unk_domains) == 0
        }

    def filter_dialog(self, data_list_file: str):
        """
        data_list_file: data list file path
        returns: dialogs in the list
        """
        with open(data_list_file, "r", encoding='utf-8') as f:
            dialog_ids = [line.strip() for line in f]
        return {
            dialog_id: dialog_content for dialog_id, dialog_content in self.full_data.items()
            if dialog_id in dialog_ids
        }

    def extract_action(self, raw_act, with_values=True):
        action = set()
        for domain_intent, slot_values in raw_act.items():
            domain, intent = domain_intent.split('-')
            for slot, value in slot_values:
                if with_values:
                    action.add((domain, intent, slot, value))
                else:
                    action.add((domain, intent, slot))
        return action

    def concatenate_history_and_utt(self, history, user_utt, raw_user_act):
        # update state
        if len(history) > 0:
            state = copy.deepcopy(history[-1]['metadata'])
            sys_act = frozenset(self.extract_action(history[-1]['dialog_act'], with_values=True))
        else:
            state = copy.deepcopy(self.void_state)
            sys_act = frozenset()
        sys_act_slot_values = defaultdict(list)
        for domain, intent, slot, value in sys_act:
            sys_act_slot_values[(domain, intent, slot)].append(value)
        for domain_intent_slot, values in sys_act_slot_values.items():
            domain, intent, slot = domain_intent_slot
            slot = slot.lower()
            if not (intent in ['Inform', 'Recommend'] and len(values) == 1):
                continue
            if slot not in ACT_SLOT_TO_STATE_SLOT or domain.lower() == 'booking':
                continue
            slot = ACT_SLOT_TO_STATE_SLOT[slot]
            if domain == 'Train' and slot == 'book_day':
                slot = 'day'
            if 'book' in slot:
                state[domain.lower()]['book'][slot.strip('book_')] = values[0]
            else:
                state[domain.lower()]['semi'][slot] = values[0]

        user_act = self.extract_action(raw_user_act)
        user_act_slot_values = defaultdict(list)
        for domain, intent, slot, value in user_act:
            user_act_slot_values[(domain, intent, slot)].append(value)
        for domain_intent_slot, values in user_act_slot_values.items():
            domain, intent, slot = domain_intent_slot
            slot = slot.lower()
            if intent != 'Inform':
                continue
            if slot not in ACT_SLOT_TO_STATE_SLOT or domain.lower() == 'booking':
                continue
            slot = ACT_SLOT_TO_STATE_SLOT[slot]
            if 'book' in slot:
                state[domain.lower()]['book'][slot.strip('book_')] = "|".join(values)
            else:
                state[domain.lower()]['semi'][slot] = "|".join(values)
        return history + [
            {'text': user_utt, 'dialog_act': raw_user_act, 'metadata': {}},
            {'text': '', 'dialog_act': {}, 'metadata': state}
        ]

    def extract_system_acts_from_dialog_data(self, data):
        acts = {}
        for dialog_id_, dialog_content in data.items():
            dialog_id = dialog_id_[:-5]  # drop .json
            for turn_id, turn in enumerate(dialog_content['log']):
                if turn_id % 2 == 0:
                    continue  # user
                if dialog_id not in acts:
                    acts[dialog_id] = {}
                if turn['dialog_act'] != {}:
                    acts[dialog_id][str(turn_id // 2 + 1)] = turn['dialog_act']
        return acts

    def save_data(self, data, dir_path):
        """
        save data to the directory dir_path, including dialogs and dialog system actions
        """
        os.makedirs(dir_path, exist_ok=True)
        print(f"saving {len(data)} dialogs into {dir_path}")
        with open(f"{dir_path}/data.json", "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        with open(f"{dir_path}/system_acts.json", "w", encoding='utf-8') as f:
            json.dump(self.extract_system_acts_from_dialog_data(data), f, indent=2)

    def is_unseen_value(self, raw_action):
        action = frozenset(self.extract_action(raw_action, with_values=True))
        for _, _, _, value in action:
            if value not in self.id_slot_values:
                return True

    def is_unseen_comb(self, raw_action):
        action = frozenset(self.extract_action(raw_action, with_values=False))
        return action not in self.id_act_comb # (not self.is_unseen_value(raw_action)) and (action not in self.id_act_comb)

    def is_locally_contextual_ood(self, system_action, raw_action):
        action = frozenset(self.extract_action(raw_action, with_values=False))
        return (not self.is_unseen_comb(raw_action))  and (action not in self.id_sys_user_map[system_action])

    def is_globally_contextual_ood_with_id_history(self, history_actions, raw_action):
        action = self.extract_action(raw_action, with_values=False)
        if len(history_actions) == 0:
            system_action = frozenset()
        else:
            system_action = history_actions[-1]
        return (not self.is_unseen_comb(raw_action)) \
               and (not action not in self.id_sys_user_map[system_action]) \
               and (action not in self.id_history_user_map[tuple(history_actions)])

    def pick_unseen_value_utt(self, dialog_history):
        """
        param dialog_history: id dialog history
        returns: ood dialog with id history and an unseen value utterance, picked (utt action)
        """
        k = random.randint(0, len(self.utterance_with_act) - 1)
        for entry in self.utterance_with_act[k:] + self.utterance_with_act[:k]:
            utterance, raw_action = entry
            if self.is_unseen_value(raw_action):
                return self.concatenate_history_and_utt(dialog_history, utterance, raw_action), entry
        return None, None

    def pick_unseen_comb_utt(self, dialog_history):
        """
        param dialog_history: id dialog history
        returns: ood dialog with id history and an unseen combination utterance, picked utt action
        """
        k = random.randint(0, len(self.utterance_with_act) - 1)
        for entry in self.utterance_with_act[k:] + self.utterance_with_act[:k]:
            utterance, raw_action = entry
            if self.is_unseen_comb(raw_action):
                return self.concatenate_history_and_utt(dialog_history, utterance, raw_action), entry
        return None, None

    def pick_locally_contextual_ood_utt(self, dialog_history):
        k = random.randint(0, len(self.utterance_with_act) - 1)
        if len(dialog_history) == 0:
            sys_act = frozenset()
        else:
            sys_act = frozenset(self.extract_action(dialog_history[-1]['dialog_act'], with_values=False))
        for entry in self.utterance_with_act[k:] + self.utterance_with_act[:k]:
            utterance, raw_action = entry
            if self.is_locally_contextual_ood(sys_act, raw_action):
                return self.concatenate_history_and_utt(dialog_history, utterance, raw_action), entry
        return None, None

    def pick_globally_contextual_ood_utt(self, dialog_history):
        history = [frozenset(self.extract_action(turn['dialog_act'], with_values=False)) for turn in dialog_history]
        k = random.randint(0, len(self.utterance_with_act) - 1)
        for entry in self.utterance_with_act[k:] + self.utterance_with_act[:k]:
            utterance, raw_action = entry
            if self.is_globally_contextual_ood_with_id_history(history, raw_action):
                return self.concatenate_history_and_utt(dialog_history, utterance, raw_action), entry
        return None, None

    def pick_ood_history_for_non_contextual_ood(self, user_utt_act, history_length=-1):
        """
        param user_utt_act: (user utterance, user action)
        returns: dialog
        """
        utterance, raw_action = user_utt_act
        k = random.randint(0, len(self.ood_dialog_history) - 1)
        if history_length == -1:
            self.concatenate_history_and_utt(self.ood_dialog_history[k], utterance, raw_action)
        else:
            for history in self.ood_dialog_history[k:] + self.ood_dialog_history[:k]:
                if len(history) == history_length:
                    return self.concatenate_history_and_utt(history, utterance, raw_action)
        return None

    def pick_ood_history_for_locally_contextual_ood(self, user_utt_act, history_length=-1):
        """
        param user_utt_act: (user utterance, user action)
        returns: dialog
        """
        k = random.randint(0, len(self.ood_dialog_history) - 1)
        utterance, raw_action = user_utt_act
        user_action = frozenset(self.extract_action(raw_action, with_values=False))
        for history in self.ood_dialog_history[k:] + self.ood_dialog_history[:k]:
            if history_length != -1 and (len(history) != history_length):
                continue
            sys_action = frozenset(self.extract_action(history[-1]['dialog_act'], with_values=False))
            if user_action not in self.id_sys_user_map[sys_action]:
                return self.concatenate_history_and_utt(history, utterance, raw_action)
        # print("failed in ood his, locally contextual ")
        return None

    def pick_ood_history_for_globally_contextual_ood(self, user_utt_act, history_length=-1):
        k = random.randint(0, len(self.ood_dialog_history) - 1)
        utterance, raw_action = user_utt_act
        user_action = frozenset(self.extract_action(raw_action, with_values=False))
        for history in self.ood_dialog_history[k:] + self.ood_dialog_history[:k]:
            if history_length != -1 and (len(history) != history_length):
                continue
            sys_action = frozenset(self.extract_action(history[-1]['dialog_act'], with_values=False))
            if user_action in self.id_sys_user_map[sys_action]:
                return self.concatenate_history_and_utt(history, utterance, raw_action)
        # print("failed in ood his, globally contextual")
        return None

    def init_ood_dialog_history(self, data):
        dialogs = []
        for dialog_id, dialog_content in data.items():
            history = []
            is_id_history = True
            for turn_id, turn in enumerate(dialog_content['log']):
                raw_current_action = copy.deepcopy(turn['dialog_act'])
                current_action = frozenset(self.extract_action(turn['dialog_act'], with_values=False))
                if turn_id % 2 == 0:  # user turn
                    if is_id_history:
                        if self.is_unseen_value(raw_current_action) or self.is_unseen_comb(raw_current_action) \
                                or (
                                len(history) > 0 and self.is_locally_contextual_ood(history[-1], raw_current_action)) \
                                or self.is_globally_contextual_ood_with_id_history(history, raw_current_action):
                            is_id_history = False
                elif not is_id_history:  # system turn
                    dialogs.append(dialog_content['log'][:turn_id + 1])
                history.append(current_action)
        return dialogs

    def split_id_ood(self, data):
        """
        param data: data to split
        returns: id dialogs,
                 id history+unseen value utt, id history+unseen comb utt, id history+locally contextual ood utt,
                 id history+globally contextual ood utt, ood history+unseen value utt, ood history+unseen comb utt,
                 ood history+locally contextual ood utt, ood history+globally contextual ood utt
        """
        id_dialogs = {}
        # id_hist_unseen_value_utt = {}
        id_hist_unseen_comb_utt = {}
        id_hist_loc_ood_utt = {}
        id_hist_glo_ood_utt = {}
        # ood_hist_unseen_value_utt = {}
        ood_hist_unseen_comb_utt = {}
        ood_hist_loc_ood_utt = {}
        ood_hist_glo_ood_utt = {}
        for dialog_id, dialog_content in data.items():
            dialog_id = dialog_id[:-5]  # drop .json
            history = []
            is_id_history = True
            for turn_id, turn in enumerate(dialog_content['log']):
                raw_current_action = copy.deepcopy(turn['dialog_act'])
                current_action = frozenset(self.extract_action(turn['dialog_act'], with_values=False))
                if turn_id % 2 == 0:
                    if is_id_history:
                        # if self.is_unseen_value(raw_current_action):
                        #    id_hist_unseen_value_utt[f"{dialog_id}-ID-UV-{turn_id}.json"] = {
                        #        'log': dialog_content['log'][:turn_id + 2]
                        #    }
                        #    is_id_history = False
                        if self.is_unseen_comb(raw_current_action):
                            id_hist_unseen_comb_utt[f"{dialog_id}-ID-UC-{turn_id}.json"] = {
                                'log': dialog_content['log'][:turn_id + 2]
                            }
                            is_id_history = False
                        elif len(history) > 0 and self.is_locally_contextual_ood(history[-1], raw_current_action):
                            id_hist_loc_ood_utt[f"{dialog_id}-ID-LC-{turn_id}.json"] = {
                                'log': dialog_content['log'][:turn_id + 2]
                            }
                            is_id_history = False
                        elif self.is_globally_contextual_ood_with_id_history(history, raw_current_action):
                            id_hist_glo_ood_utt[f"{dialog_id}-ID-GC-{turn_id}.json"] = {
                                'log': dialog_content['log'][:turn_id + 2]
                            }
                            is_id_history = False
                        else:
                            id_dialogs[f"{dialog_id}-ID-ID-{turn_id}.json"] = {
                                'log': dialog_content['log'][:turn_id + 2]
                            }
                    else:
                        # if self.is_unseen_value(raw_current_action):
                        #   ood_hist_unseen_value_utt[f"{dialog_id}-OoD-UV-{turn_id}.json"] = {
                        #         'log': dialog_content['log'][:turn_id + 2]
                        #     }
                        if self.is_unseen_comb(raw_current_action):
                            ood_hist_unseen_comb_utt[f"{dialog_id}-OoD-UC-{turn_id}.json"] = {
                                'log': dialog_content['log'][:turn_id + 2]
                            }
                        elif len(history) > 0 and self.is_locally_contextual_ood(history[-1], raw_current_action):
                            ood_hist_loc_ood_utt[f"{dialog_id}-OoD-LC-{turn_id}.json"] = {
                                'log': dialog_content['log'][:turn_id + 2]
                            }
                        else:
                            ood_hist_glo_ood_utt[f"{dialog_id}-OoD-GC-{turn_id}.json"] = {
                                'log': dialog_content['log'][:turn_id + 2]
                            }
                history.append(current_action)
        return {
            "id_dialogs": id_dialogs, 
            "id_hist_non_contextual_ood_utt": id_hist_unseen_comb_utt, 
            "id_hist_contextual_ood_utt": {**id_hist_loc_ood_utt, **id_hist_glo_ood_utt},
            "ood_hist_non_contextual_ood_utt": ood_hist_unseen_comb_utt,
            "ood_hist_contextual_ood_utt": {**ood_hist_loc_ood_utt, **ood_hist_glo_ood_utt}
        }

    def generate_ood(self, data):
        """
        param data: data, including id and ood, need to filter out ood
        returns id dialogs,
                 id history+unseen value utt, id history+unseen comb utt, id history+locally contextual ood utt,
                 id history+globally contextual ood utt, ood history+unseen value utt, ood history+unseen comb utt,
                 ood history+locally contextual ood utt, ood history+globally contextual ood utt
        ID(H1, u1) -> ID His+OoD Utt(H1, u2) -> OoD His+OoD Utt(H2, u2)
        """
        id_dialogs = {}
        # id_hist_unseen_value_utt = {}
        id_hist_unseen_comb_utt = {}
        id_hist_loc_ood_utt = {}
        id_hist_glo_ood_utt = {}
        # ood_hist_unseen_value_utt = {}
        ood_hist_unseen_comb_utt = {}
        ood_hist_loc_ood_utt = {}
        ood_hist_glo_ood_utt = {}
        for dialog_id, dialog_content in data.items():
            dialog_id = dialog_id[:-5]  # drop .json
            history = []
            for turn_id, turn in enumerate(dialog_content['log']):
                raw_current_action = copy.deepcopy(turn['dialog_act'])
                current_action = frozenset(self.extract_action(turn['dialog_act'], with_values=False))
                if turn_id % 2 == 0:
                    if  self.is_unseen_comb(raw_current_action) \
                            or (len(history) > 0 and self.is_locally_contextual_ood(history[-1], raw_current_action)) \
                            or self.is_globally_contextual_ood_with_id_history(history, raw_current_action):
                        break
                    # id_hist_unseen_val_utt_dialog, unseen_value_action = \
                    #     self.pick_unseen_value_utt(dialog_content['log'][:turn_id])
                    id_hist_unseen_comb_utt_dialog, unseen_comb_action = \
                        self.pick_unseen_comb_utt(dialog_content['log'][:turn_id])
                    id_hist_loc_ood_utt_dialog, loc_ood_action = \
                        self.pick_locally_contextual_ood_utt(dialog_content['log'][:turn_id])
                    id_hist_glo_ood_utt_dialog, glo_ood_action = \
                        self.pick_globally_contextual_ood_utt(dialog_content['log'][:turn_id])
                    if (unseen_comb_action is not None) \
                            and (loc_ood_action is not None) and (glo_ood_action is not None):
                        # ood_hist_unseen_val_utt_dialog = \
                        #     self.pick_ood_history_for_non_contextual_ood(unseen_value_action, len(history))
                        ood_hist_unseen_comb_utt_dialog = \
                            self.pick_ood_history_for_non_contextual_ood(unseen_comb_action, len(history))
                        ood_hist_loc_ood_utt_dialog = \
                            self.pick_ood_history_for_locally_contextual_ood(loc_ood_action, len(history))
                        ood_hist_glo_ood_utt_dialog = \
                            self.pick_ood_history_for_globally_contextual_ood(glo_ood_action, len(history))
                        if (ood_hist_unseen_comb_utt_dialog is not None) \
                                and (ood_hist_loc_ood_utt_dialog is not None) \
                                and (ood_hist_glo_ood_utt_dialog is not None):
                            id_dialogs[f"{dialog_id}-ID-ID-{turn_id}.json"] = {
                                'log': dialog_content['log'][:turn_id + 2]
                            }
                            # id_hist_unseen_value_utt[f"{dialog_id}-ID-UV-{turn_id}.json"] = {
                            #     'log': id_hist_unseen_val_utt_dialog
                            # }
                            id_hist_unseen_comb_utt[f"{dialog_id}-ID-UC-{turn_id}.json"] = {
                                'log': id_hist_unseen_comb_utt_dialog
                            }
                            id_hist_loc_ood_utt[f"{dialog_id}-ID-LC-{turn_id}.json"] = {
                                'log': id_hist_loc_ood_utt_dialog
                            }
                            id_hist_glo_ood_utt[f"{dialog_id}-ID-GC-{turn_id}.json"] = {
                                'log': id_hist_glo_ood_utt_dialog
                            }
                            # ood_hist_unseen_value_utt[f"{dialog_id}-OoD-UV-{turn_id}.json"] = {
                            #     'log': ood_hist_unseen_val_utt_dialog
                            # }
                            ood_hist_unseen_comb_utt[f"{dialog_id}-OoD-UC-{turn_id}.json"] = {
                                'log': ood_hist_unseen_comb_utt_dialog
                            }
                            ood_hist_loc_ood_utt[f"{dialog_id}-OoD-LC-{turn_id}.json"] = {
                                'log': ood_hist_loc_ood_utt_dialog
                            }
                            ood_hist_glo_ood_utt[f"{dialog_id}-OoD-GC-{turn_id}.json"] = {
                                'log': ood_hist_glo_ood_utt_dialog
                            }
                history.append(current_action)

        return {
            "id_dialogs": id_dialogs, 
            "id_hist_non_contextual_ood_utt": id_hist_unseen_comb_utt, 
            "id_hist_contextual_ood_utt": {**id_hist_loc_ood_utt, **id_hist_glo_ood_utt},
            "ood_hist_non_contextual_ood_utt": ood_hist_unseen_comb_utt,
            "ood_hist_contextual_ood_utt": {**ood_hist_loc_ood_utt, **ood_hist_glo_ood_utt}
        }


    def run(self, data_home_dir):
        print("saving training data...")
        os.makedirs(f"{data_home_dir}/train", exist_ok=True)
        with open(f'{data_home_dir}/train/train.json', "w", encoding='utf-8') as f:
            json.dump(self.ori_train_data, f, indent=2)
        with open(f'{data_home_dir}/train/train_dialog_acts.json', "w", encoding='utf-8') as f:
            json.dump(self.extract_system_acts_from_dialog_data(self.ori_train_data), f, indent=2)
        with open(f'{data_home_dir}/train/valid.json', "w", encoding='utf-8') as f:
            json.dump(self.ori_valid_data, f, indent=2)
        with open(f'{data_home_dir}/train/valid_dialog_acts.json', "w", encoding='utf-8') as f:
            json.dump(self.extract_system_acts_from_dialog_data(self.ori_valid_data), f, indent=2)
        print("training data saved\n")

        print("splitting test data into id and ood...")
        test_data_splits = self.split_id_ood(self.ori_test_data)
        print("original test data splitted\n")
        print("generating experiment test data...")
        generated_data_splits = self.generate_ood(self.ori_test_data)
 
        for dialog_type, dialogs in test_data_splits.items():
            self.save_data(dialogs, f"{data_home_dir}/test_{dialog_type}")
        for dialog_type, dialogs in generated_data_splits.items():
            self.save_data(dialogs, f"{data_home_dir}/generated_{dialog_type}") 

        print("experiment data generated\n")

        print("OoD data prepared.\n")
