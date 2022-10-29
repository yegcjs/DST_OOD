import abc
import multiprocessing as mp

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def parallel_tokenize(tokenizer, texts, collate_fn):
    chunks = [texts[i:i + 1024] for i in range(0, len(texts), 1024)]
    tokenizer(chunks[0])
    # os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    with mp.Pool(min(mp.cpu_count(), len(chunks))) as p:
        tokenized_texts = list(tqdm(p.imap(tokenizer, chunks), total=len(chunks)))
    return collate_fn(tokenized_texts)


class DSTDataManager(abc.ABC):
    def __init__(self, tokenizer, debug):
        super(DSTDataManager, self).__init__()
        self.tokenizer = tokenizer
        self.datasets = {}
        self.debug = debug

    @staticmethod
    @abc.abstractmethod
    def collate_fn(batch):
        pass

    @abc.abstractmethod
    def load_dataset(self, data_split, path, device):
        pass

    def get_loader(self, data_split, batch_size, shuffle=True, distributed_world_size=1, distributed_rank=-1):
        if distributed_world_size == 1:  # not distributed
            return DataLoader(
                dataset=self.datasets[data_split], batch_size=batch_size,
                shuffle=shuffle, collate_fn=self.collate_fn
            )
        else:
            sampler = DistributedSampler(
                dataset=self.datasets[data_split], num_replicas=distributed_world_size, rank=distributed_rank
            )
            return DataLoader(
                dataset=self.datasets[data_split], batch_size=batch_size, shuffle=False,
                sampler=sampler, collate_fn=self.collate_fn, num_workers=0
            )
