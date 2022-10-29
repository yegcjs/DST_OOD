from .Trainer import TrippyTrainer
from .Data import TrippyDataManager
from .Evaluator import TrippyEvaluator
from transformers import BertTokenizerFast
from .Model import TrippyModel
from tqdm import tqdm
from utils.DSTMethod import *
from utils.DSTTester import *
import torch


class Trippy(DSTMethod):
    def __init__(self, args):
        super(Trippy, self).__init__(args)

    def prepare(self):
        if self.args.checkpoint is None:
            tokenizer = BertTokenizerFast.from_pretrained(self.args.pretrained)
        else:
            tokenizer = BertTokenizerFast.from_pretrained(self.args.checkpoint)
        data = TrippyDataManager(tokenizer, self.args.config, debug=self.args.debug)
        model = TrippyModel(self.args.pretrained, data.config, self.args.save_memory)
        if self.args.checkpoint is not None:
            model.load_state_dict(torch.load(f"{self.args.checkpoint}/model.pt", map_location='cpu'))
        return tokenizer, data, model

    def train(self):
        tokenizer, data, model = self.prepare()
        tokenizer.save_pretrained(self.args.saveto)
        trainer = TrippyTrainer(
            TrippyEvaluator(config=data.config, tokenizer=tokenizer),
            self.args
        )
        trainer.train(model, data)

    def test(self):
        tokenizer, data, model = self.prepare()
        DSTTester(TrippyEvaluator(config=data.config, tokenizer=tokenizer, dump_path=self.args.dump_path), self.args).test(model, data)
