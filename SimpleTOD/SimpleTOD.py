from .Trainer import SimpleTODTrainer
from .Data import SimpleTODDataManager
from .Evaluator import SimpleTODEvaluator
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from tqdm import tqdm
from utils.DSTMethod import *
from utils.DSTTester import *


class SimpleTOD(DSTMethod):
    def __init__(self, args):
        super(SimpleTOD, self).__init__(args)

    def prepare(self):
        if self.args.checkpoint is None:
            tokenizer = GPT2TokenizerFast.from_pretrained(self.args.pretrained)
            special_tokens = {
                'additional_special_tokens':
                    ["<|user|>", "<|system|>", "<|context|>", "<|endofcontext|>", "<|belief|>", "<|endofbelief|>"]
            }
            tokenizer.add_special_tokens(special_tokens)
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained(self.args.pretrained)
            model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer = GPT2TokenizerFast.from_pretrained(self.args.checkpoint)
            model = GPT2LMHeadModel.from_pretrained(self.args.checkpoint)

        data = SimpleTODDataManager(tokenizer, debug=self.args.debug)

        return tokenizer, data, model

    def train(self):
        tokenizer, data, model = self.prepare()
        tokenizer.save_pretrained(self.args.saveto)
        trainer = SimpleTODTrainer(
            SimpleTODEvaluator(tokenizer=tokenizer),
            self.args
        )
        trainer.train(model, data)

    def test(self):
        tokenizer, data, model = self.prepare()
        DSTTester(SimpleTODEvaluator(tokenizer=tokenizer, dump_path=self.args.dump_path), self.args).test(model, data)
