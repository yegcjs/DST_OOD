from copy import deepcopy
import fitlog
from torch.optim import AdamW
import torch
from utils.MPDSTTrainer import MPDSTTrainer
from torch_poly_lr_decay import PolynomialLRDecay

class TrippyTrainer(MPDSTTrainer):
    def __init__(self, evaluator, args):
        super(TrippyTrainer, self).__init__(evaluator, args)
        self.lambda_gate = args.lambda_gate
        self.lambda_span = args.lambda_span
        self.lambda_refer = args.lambda_refer

    def init_optimizer(self, model):
        optimizer = AdamW([
            {'params': model.bert.parameters(), 'lr': 2e-5},
            {'params': model.gates.parameters(), 'lr': 1e-3},
            {'params': model.span.parameters(), 'lr': 1e-3},
            {'params': model.refer.parameters(), 'lr': 1e-3}
        ])
        # scheduler = PolynomialLRDecay(optimizer, max_decay_steps=1000, end_learning_rate=5e-5, power=2.0)
        return optimizer

    def get_batch_loss(self, model, batch):
        text_ids, text_masks, token_type_ids, states, informs, gates, span_start, span_end, ref_slot, _ = batch
        model_outputs = model(
            input_ids=text_ids,
            attention_mask=text_masks,
            token_type_ids=token_type_ids,
            states=states,
            informs=informs,
            gates=gates,
            span_start=span_start,
            span_end=span_end,
            ref_slot=ref_slot
        )
        return self.lambda_gate * model_outputs.gate_loss \
               + self.lambda_span * model_outputs.span_loss \
               + self.lambda_refer * model_outputs.span_loss

    # def save_best_model(self, model, metric, path):
    #     """
    #     :return: True -> continue, False -> end training
    #     """
    #     if metric >= self.best_eval_acc:
    #         self.best_eval_acc = deepcopy(metric)
    #         fitlog.add_best_metric({'eval': {'joint-acc': metric}})
    #         torch.save(model.state_dict(), f"{path}/model.pt")
    #         self.over_fitting_count = 0
    #     else:
    #         self.over_fitting_count += 1
    #     return self.over_fitting_count <= self.args.early_stop
