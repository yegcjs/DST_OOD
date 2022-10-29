import pdb
import json 
import torch
from utils.DSTEvaluator import *
from tqdm import tqdm

class SimpleTODEvaluator(DSTEvaluator):
    def __init__(self, tokenizer, dump_path=None):
        super(SimpleTODEvaluator, self).__init__(dump_path)
        self.tokenizer = tokenizer

    def state_diff(self, prev_state, current_state):
        turn_state = set()
        for item in current_state:
            if item not in prev_state:
                turn_state.add(item)
        return turn_state

    @torch.no_grad()
    def evaluate(self, model, dataloader, only_last=False):
        dump_info = []
        model = model.eval()
        ending_token_id = {self.tokenizer('<|endofbelief|>')['input_ids'][0], self.tokenizer.eos_token_id}
        context_start_id, context_end_id = self.tokenizer('<|context|>')['input_ids'][0], \
                                           self.tokenizer('<|endofcontext|>')['input_ids'][0],
        previous_his, prev_result, prev_turn_result = "!@#$%^&*()_", 0, 0
        num_correct, turn_correct, total = 0, 0, 0
        for _, _, histories, target_states in tqdm(dataloader):
            for history, target_state in zip(histories, target_states):
                history = history.unsqueeze(0)

                his = history[0].tolist()
                curr = "-".join(map(str, his[his.index(context_start_id) + 1: his.index(context_end_id)]))
                first_turn = (previous_his not in curr)
                if first_turn:
                    prev_predicted_state, prev_oracle_state = set(), set()
                previous_his = curr

                output = model(input_ids=history, use_cache=True, return_dict=True)
                cache = output.past_key_values
                predicted = torch.argmax(output.logits[0, -1, :])
                generating_state = self.tokenizer.decode(predicted.view(1)).strip(' ,')
                predicted_state = set()
                for _ in range(1024 - history.shape[-1]):  # max sequence length is 1024
                    if predicted.item() in ending_token_id:
                        break
                    output = model(input_ids=predicted.view(1, 1), past_key_values=cache, use_cache=True,
                                    return_dict=True)
                    cache, predicted = output.past_key_values, torch.argmax(output.logits[0, -1, :])
                    generated_token = self.tokenizer.decode(predicted.view(1))
                    # print(generated_token)
                    if generated_token in {',', '<|endofbelief|>'}:
                        generating_state = generating_state.strip()
                        # print(generating_state)
                        if ('none' in generating_state) or ('not mentioned' in generating_state):
                            generating_state = ""
                        # elif generating_state not in target_state:
                        #     pass
                        else:
                            predicted_state.add(generating_state)
                            generating_state = ""
                    else:
                        generating_state += generated_token

                if only_last:
                    if not first_turn:
                        total -= 1
                        num_correct -= prev_result
                        turn_correct -= prev_turn_result

                predicted_turn_state = self.state_diff(prev_predicted_state, predicted_state)
                oracle_turn_state = self.state_diff(prev_oracle_state, target_state)
                prev_result = 1 if predicted_state == target_state else 0
                prev_turn_result = 1 if predicted_turn_state == oracle_turn_state else 0
                num_correct += prev_result
                turn_correct += prev_turn_result
                total += 1

                if self.dump_path is not None and prev_result==1 and prev_turn_result==0:
                    dump_info.append(
                        {
                            'text': self.tokenizer.decode(his),
                            'prev_state': list(prev_predicted_state),
                            'state': list(predicted_state),
                            'prev_oracle_state': list(prev_oracle_state),
                            'oracle_state': list(target_state)
                        }
                    )
                prev_predicted_state, prev_oracle_state = predicted_state, target_state

        if self.dump_path is not None:
            with open(self.dump_path, "w", encoding='utf-8') as f: 
                json.dump(dump_info, f, indent=2)

        return float(num_correct) / float(total), float(turn_correct) / float(total)
