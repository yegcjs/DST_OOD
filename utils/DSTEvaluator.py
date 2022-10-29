import abc
import os 
import torch


class DSTEvaluator(abc.ABC):
    def __init__(self, dump_path=None):
        super().__init__()
        self.dump_path = dump_path
        if dump_path is not None:
            directory = '/'.join(dump_path.rstrip('/').split('/')[:-1])
            os.makedirs(directory, exist_ok=True)

    @abc.abstractmethod
    @torch.no_grad()
    def evaluate(self, model, dataloader, repetition_check_enable=False):
        pass
