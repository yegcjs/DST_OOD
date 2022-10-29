import json
import pdb
import torch
from utils.DSTEvaluator import DSTEvaluator
from tqdm import tqdm
from transformers import BertTokenizer
from copy import deepcopy


class TrippyEvaluator(DSTEvaluator):
    def __init__(self, config, tokenizer, dump_path=None):
        super(TrippyEvaluator, self).__init__(dump_path)
        self.config = config  # idx2label, idx2slot, ect.
        self.tokenizer = tokenizer

    def state_diff(self, previous_state, current_state):
        turn_state = {}
        for slot, value in current_state.items():
            encode_decode_cur_value = self.tokenizer.decode(self.tokenizer.encode(value)[1:-1])
            encode_decode_prev_value = self.tokenizer.decode(self.tokenizer.encode(previous_state[slot])[1:-1])
            if encode_decode_prev_value == encode_decode_cur_value:
                turn_state[slot] = "none"
                continue
            elif value in self.config.value_maps:
                encode_decode_cur_variants = [
                    self.tokenizer.decode(self.tokenizer.encode(variant)[1:-1])
                    for variant in self.config.value_maps[value]
                ]
                if encode_decode_prev_value in encode_decode_cur_variants:
                    turn_state[slot] = "none"
                    continue
            elif previous_state[slot] in self.config.value_maps:
                encode_decode_prev_variants = [
                    self.tokenizer.decode(self.tokenizer.encode(variant)[1:-1])
                    for variant in self.config.value_maps[previous_state[slot]]
                ]
                if encode_decode_cur_value in encode_decode_prev_variants:
                    turn_state[slot] = "none"
                    continue
            turn_state[slot] = value
        return turn_state

    def equal_state(self, prediction_state, target_state):
        for slot in self.config.idx2slot:
            encode_decode_target_value = self.tokenizer.decode(self.tokenizer.encode(target_state[slot])[1:-1])
            if prediction_state[slot] == encode_decode_target_value:
                continue
            if target_state[slot] in self.config.value_maps:
                encode_decode_target_variants = [
                    self.tokenizer.decode(self.tokenizer.encode(variant)[1:-1])
                    for variant in self.config.value_maps[target_state[slot]]
                ]
                if prediction_state[slot] in encode_decode_target_variants:
                    continue
            return False
        return True

    @torch.no_grad()
    def evaluate(self, model, dataloader, only_last=False):
        dump_info = []

        correct_cnt, tot = 0, 0
        turn_correct_cnt = 0
        prev_res, prev_turn_res = 0, 0
        model.eval()
        for batch in tqdm(dataloader):
            text_ids, text_masks, token_type_ids, oracle_known_states, informs, \
            gates, span_start, span_end, ref_slots, target_states = batch
            batch_size = text_ids.shape[0]
            for i in range(batch_size):
                mask_list = text_masks[i].tolist()
                eos_pos = mask_list.index(0) - 1 if mask_list[-1] == 0 else -1
                first_turn = text_ids[i][eos_pos - 1].item() == self.tokenizer.sep_token_id
                oracle_state = target_states[i]
                if first_turn:  # history == ""
                    known_state = torch.zeros_like(oracle_known_states[i])
                    oracle_prev_state = {slot: "none" for slot in self.config.idx2slot}
                    state = {slot: "none" for slot in self.config.idx2slot}
                prev_state = deepcopy(state)
                model_outputs = model(
                    input_ids=text_ids[i].unsqueeze(0),
                    attention_mask=text_masks[i].unsqueeze(0),
                    token_type_ids=token_type_ids[i].unsqueeze(0),
                    states=known_state.unsqueeze(0),
                    informs=informs[i].unsqueeze(0)
                )
                gate_predictions = model_outputs.gate_logits.argmax(dim=-1).squeeze()  # num_slots
                start_predictions = model_outputs.span_logits[:, :, :, 0].argmax(dim=-1).squeeze()  # num_slots
                end_predictions = model_outputs.span_logits[:, :, :, 1].argmax(dim=-1).squeeze()  # num_slots
                ref_predictions = model_outputs.ref_logits.argmax(dim=-1).squeeze()  # num_slots
                for slot_id, gate_prediction in enumerate(gate_predictions):
                    gate_prediction = gate_prediction.item()
                    if self.config.idx2label[gate_prediction] == 'none':
                        continue
                    else:
                        known_state[slot_id] = 1
                        if self.config.idx2label[gate_prediction] in ['true', 'false', 'dontcare']:
                            state[self.config.idx2slot[slot_id]] = self.config.idx2label[gate_prediction]
                        elif self.config.idx2label[gate_prediction] == 'copy_value':
                            start, end = start_predictions[slot_id], end_predictions[slot_id] + 1
                            span_value = self.tokenizer.decode(text_ids[i][start: end])
                            state[self.config.idx2slot[slot_id]] = span_value
                        elif self.config.idx2label[gate_prediction] == 'inform':
                            state[self.config.idx2slot[slot_id]] = target_states[i][self.config.idx2slot[slot_id]]
                        else:
                            ref_slot_id = ref_predictions[slot_id]
                            state[self.config.idx2slot[slot_id]] = state[self.config.idx2slot[ref_slot_id]]

                if only_last:
                    if not first_turn:
                        tot -= 1
                        correct_cnt -= prev_res
                        turn_correct_cnt -= prev_turn_res
                    # key = "-".join(map(str, text_ids[i].tolist()))
                    # if key in known_inputs:
                    #     continue
                    # else:
                    #     known_inputs.add(key)
                prev_res = 1 if self.equal_state(state, target_states[i]) else 0
                prev_turn_res = 1 if self.equal_state(
                    self.state_diff(prev_state, state),
                    self.state_diff(oracle_prev_state, oracle_state)
                ) else 0
                if self.dump_path is not None and prev_res == 1 and prev_turn_res == 0:#  and self.equal_state(prev_state, oracle_prev_state):
                    dump_info.append(
                        {
                            'text': self.tokenizer.decode(text_ids[i]),
                            'prev_state': prev_state,
                            'current_state': state,
                            'prev_oracle_state': oracle_prev_state,
                            'oracle_state': oracle_state
                        }
                    )
                    # print("P") # problematic!
                oracle_prev_state = oracle_state
                correct_cnt += prev_res
                turn_correct_cnt += prev_turn_res
                tot = tot + 1

        if self.dump_path is not None:
            with open(self.dump_path, "w", encoding='utf-8') as f: 
                json.dump(dump_info, f, indent=2)
        print(correct_cnt, turn_correct_cnt, tot)
        return float(correct_cnt) / float(tot), float(turn_correct_cnt) / float(tot)
