import pdb

import torch.multiprocessing as mp
# from .fitlog_utils import *
import torch.distributed as dist
import abc
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy

torch.multiprocessing.set_sharing_strategy('file_system')


class MPDSTTrainer(abc.ABC):
    def __init__(self, evaluator, args):
        self.evaluator = evaluator
        self.args = args
        self.best_eval_acc = -1
        self.over_fitting_count = 0

    @abc.abstractmethod
    def init_optimizer(self, model):
        pass

    @abc.abstractmethod
    def get_batch_loss(self, model, batch):
        pass

    def save_best_model(self, model, metric, path):
        if metric >= self.best_eval_acc:
            self.best_eval_acc = deepcopy(metric)
            # fitlog.add_best_metric({'eval': {'joint-acc': metric}})
            torch.save(model.state_dict(), f"{path}/model.pt")
            self.over_fitting_count = 0
        else:
            self.over_fitting_count += 1
        if metric < 0.4:    # avoid too early stop
            return True
        return self.over_fitting_count <= self.args.early_stop

    def run_training(self, rank, model, optimizer, train_loader, valid_loader, args):
        step, train_loss = 0, 0
        for epoch in range(args.epochs):
            if rank == 0:
                print(f"Epoch {epoch}:")
                train_loader = tqdm(train_loader)
            for batch in train_loader:
                step += 1
                with model.no_sync():
                    model.train()
                    loss = self.get_batch_loss(model, batch)
                    loss = loss / args.update_steps
                    loss.backward()
                    train_loss += loss.item() / args.eval_steps

                if step % args.update_steps == 0:
                    for p in model.parameters():
                        if p.grad is None:
                            p.grad = torch.zeros_like(p, requires_grad=False)
                        dist.all_reduce(p.grad)
                    for p in model.buffers():
                        dist.broadcast(p.data, 0)
                    optimizer.step()
                    optimizer.zero_grad()

                if rank == 0 and step % args.eval_steps == 0:
                    joint_acc = self.evaluator.evaluate(model, valid_loader) \
                        if step // args.eval_steps >= args.first_eval else 0
                    # fitlog.add_metric(joint_acc, name="eval_joint-acc", step=step)
                    cont_flag = self.save_best_model(model.module, joint_acc, args.saveto)
                    print(f"step {step} | TRAIN-LOSS {train_loss} | VALID-JGA {joint_acc}")
                    # fitlog.add_loss(train_loss, name='train_loss', step=step)
                    train_loss = 0
                    if not cont_flag:
                        return

    def _train(self, rank, model, data):
        args = self.args
        world_size = len(self.args.devices)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        if rank == 0:
            # init_fitlog(self.args)
            data.load_dataset('valid', f"{args.train}/valid.json", args.devices[rank])
        data.load_dataset('train', f"{args.train}/train.json", args.devices[rank])

        model = model.to(args.devices[rank])
        optimizer = self.init_optimizer(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.devices[rank]])
        train_loader = data.get_loader(
            data_split='train', batch_size=args.batch_size, shuffle=True,
            distributed_world_size=len(args.devices), distributed_rank=rank
        )
        if rank == 0:
            # train_loader = tqdm(train_loader)
            valid_loader = data.get_loader(
                data_split='valid', batch_size=args.batch_size, shuffle=False,
                distributed_world_size=1, distributed_rank=-1
            )
        else:
            valid_loader = None

        self.run_training(rank, model, optimizer, train_loader, valid_loader, args)

        # if rank == 0:
        #     fitlog.finish()

    def train_on_one(self, model, data):
        args = self.args
        # init_fitlog(self.args)
        data.load_dataset('valid', f"{args.train}/valid.json", args.devices[0])
        data.load_dataset('train', f"{args.train}/train.json", args.devices[0])
        model = model.to(args.devices[0])
        optimizer = self.init_optimizer(model)
        train_loader = data.get_loader(
            data_split='train', batch_size=args.batch_size, shuffle=True,
            distributed_world_size=1, distributed_rank=-1
        )
        valid_loader = data.get_loader(
            # data_split='train', batch_size=args.batch_size, shuffle=False,
            data_split='valid', batch_size=args.batch_size, shuffle=False,
            distributed_world_size=1, distributed_rank=-1
        )
        step, train_loss = 0, 0
        for epoch in range(args.epochs):
            print(f"Epoch {epoch}:")
            for batch in tqdm(train_loader):
                step += 1
                model.train()
                loss = self.get_batch_loss(model, batch) / args.update_steps
                loss.backward()
                train_loss += loss.item() / args.eval_steps

                if step % args.update_steps == 0:
                    optimizer.step()
                    # scheduler.step()
                    optimizer.zero_grad()
                if step % args.eval_steps == 0:
                    if step // args.eval_steps <= args.first_eval:
                        joint_acc, turn_acc = 0, 0
                    else:
                        joint_acc, turn_acc = self.evaluator.evaluate(model, valid_loader)
                    # print(joint_acc)
                    # fitlog.add_metric(joint_acc, name="eval_joint-acc", step=step)
                    # fitlog.add_metric(turn_acc, name="eval_turn-acc", step=step)
                    cont_flag = self.save_best_model(model, joint_acc, args.saveto)
                    # fitlog.add_loss(train_loss, name='train_loss', step=step)
                    print(f"step {step} | TRAIN-LOSS {train_loss} | VALID-JGA {joint_acc}")
                    train_loss = 0
                    if not cont_flag:
                        # fitlog.finish()
                        return
        # fitlog.finish()
        return

    def train(self, model, data):
        if len(self.args.devices) > 1:
            self.best_eval_acc = -1
            self.over_fitting_count = 0
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = self.args.port  # '34515'
            mp.spawn(self._train, nprocs=len(self.args.devices), args=(model, data))
        else:
            self.train_on_one(model, data)
