# from .fitlog_utils import *


class DSTTester:
    def __init__(self, evaluator, args):
        self.evaluator = evaluator
        self.args = args

    def test(self, model, data):
        model = model.to(self.args.devices[0])
        data.load_dataset(
            'test', self.args.test, self.args.devices[0]
        )
        test_data = data.get_loader(
            'test', batch_size=self.args.batch_size,
            shuffle=False, distributed_world_size=1, distributed_rank=-1
        )
        # init_fitlog(self.args)
        joint_acc, turn_acc = self.evaluator.evaluate(model, test_data, only_last=self.args.ood)
        print("tested file: ", self.args.test)
        print(f"JGA: {joint_acc}")
        print(f"TSA: {turn_acc}")
        # fitlog.add_best_metric({'test': {'joint_accuracy': joint_acc, 'turn_acc': turn_acc}})
        # fitlog.finish()
