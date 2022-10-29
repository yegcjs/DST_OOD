from utils.arguments import *
from SimpleTOD.SimpleTOD import SimpleTOD
from Trippy.Trippy import Trippy
from Trade.Trade import Trade
methods = {
    'simpletod': SimpleTOD,
    'trippy': Trippy,
    'trade': Trade
}


def main():
    args = parse_args()
    method = methods[args.method](args)
    if args.train is not None:
        method.train()
    if args.test is not None:
        method.test()


if __name__ == '__main__':
    main()
