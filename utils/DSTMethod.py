import abc
import os


class DSTMethod(abc.ABC):
    def __init__(self, args):
        if args.devices is None:
            args.devices = ['cpu']
        self.args = args
        if args.log is not None:
            self.ensure_path_exists(args.log)
        if args.saveto is not None:
            self.ensure_path_exists(args.saveto)

    @staticmethod
    def ensure_path_exists(full_path):
        path_split = full_path.split('/')
        for i in range(len(path_split)):
            path = "/".join(path_split[:i + 1])
            if not os.path.exists(path):
                os.mkdir(path)

    @abc.abstractmethod
    def prepare(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def test(self):
        pass
