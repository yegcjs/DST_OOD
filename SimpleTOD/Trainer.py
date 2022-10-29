import fitlog
from torch.optim import AdamW
from copy import deepcopy
from utils.MPDSTTrainer import MPDSTTrainer
from torch_poly_lr_decay import PolynomialLRDecay


class SimpleTODTrainer(MPDSTTrainer):
    def __init__(self, evaluator, args):
        super(SimpleTODTrainer, self).__init__(evaluator, args)

    def init_optimizer(self, model):
        optimizer = AdamW(model.parameters(), lr=2e-5)
        # scheduler = PolynomialLRDecay(optimizer, max_decay_steps=1000, end_learning_rate=1e-5, power=2.0)
        return optimizer

    def get_batch_loss(self, model, batch):
        input_ids, masks, _, _ = batch
        output = model(input_ids=input_ids, attention_mask=masks, labels=input_ids, return_dict=True)
        return output.loss

    def save_best_model(self, model, metric, path):
        """
        :return: True -> continue, False -> end training
        """
        if metric >= self.best_eval_acc:
            fitlog.add_best_metric({'eval': {'joint-acc': metric}})
            self.best_eval_acc = deepcopy(metric)
            model.save_pretrained(path)
            self.over_fitting_count = 0
        else:
            self.over_fitting_count += 1
        return self.over_fitting_count <= self.args.early_stop
