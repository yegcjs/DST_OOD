import os
from tqdm import tqdm
from utils.DSTMethod import *
from utils.DSTTester import *
import torch
from .trade_utils.config import *
from .models.TRADE import *
from .trade_utils.utils_multiWOZ_DST import prepare_data_seq


class Trade(DSTMethod):
    def __init__(self, args):
        super(Trade, self).__init__(args)
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(list(map(str, args.devices)))
        self.args.path = f"{self.args.saveto}/save" if self.args.checkpoint is None else f"{self.args.checkpoint}/save"
        self.args.decoder = 'TRADE'
        self.args.learn = 0.001
        self.args.drop = 0.2
        self.args.load_embedding = 1
        self.args.batch = self.args.batch_size

    def prepare(self):
        train_file = None if self.args.train is None else f"{self.args.train}/train.json"
        valid_file = None if self.args.train is None else f"{self.args.train}/valid.json"
        test_file = self.args.test
        train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = \
            prepare_data_seq(self.args.train is not None,
                             train_file,
                             valid_file,
                             test_file,
                             self.args,
                             self.args.task,
                             False,
                             batch_size=self.args.batch_size
            )
        model = globals()[self.args.decoder](
            hidden_size=int(self.args.hidden),
            lang=lang,
            path=self.args.checkpoint,
            task=self.args.task,
            lr=float(self.args.learn),
            dropout=float(self.args.drop),
            slots=SLOTS_LIST,
            gating_dict=gating_dict,
            args=self.args,
            nb_train_vocab=max_word
        )
        return model, train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word

    def train(self):
        model, train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = self.prepare()
        avg_best, cnt, acc = 0.0, 0, 0.0
        global_step = 0
        init_fitlog(self.args)
        for epoch in range(200):
            print("Epoch:{}".format(epoch))

            # Run the train function
            pbar = tqdm(enumerate(train), total=len(train))
            for i, data in pbar:
                global_step += 1
                model.train_batch(data, int(self.args.clip), SLOTS_LIST[1], reset=(i == 0))
                model.optimize(self.args.clip)
                avg_loss, loss_str = model.print_loss()
                fitlog.add_loss(avg_loss, name='train_loss', step=global_step)
                pbar.set_description(loss_str)
                # print(data)
                # exit(1)

                if (global_step + 1) % self.args.eval_steps == 0:

                    acc = model.evaluate(dev, SLOTS_LIST[2], 'JGA')
                    model.scheduler.step(acc)
                    fitlog.add_metric(acc, name="eval_joint-acc", step=global_step)
                    if acc >= avg_best:
                        avg_best = acc
                        cnt = 0
                        best_model = model
                        best_model.save_model(self.args.saveto)

                    else:
                        cnt += 1

                    if cnt == self.args.early_stop or (acc == 1.0):
                        print("Ran out of patient, early stop...")
                        fitlog.finish()
                        return
        fitlog.finish()

    def test(self):
        model, train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = self.prepare()
        init_fitlog(self.args)
        joint_acc_test, turn_acc_test = model.evaluate(test, SLOTS_LIST[3], 'JGA', last_only=self.args.ood)
        fitlog.add_best_metric({'test': {'joint_accuracy': joint_acc_test, 'turn_acc': turn_acc_test}})
        fitlog.finish()
