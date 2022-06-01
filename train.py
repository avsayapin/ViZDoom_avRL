import sys

from sample_factory.envs.doom.doom_utils import make_doom_env
# from sample_factory.algorithms.utils.arguments import default_cfg
from sample_factory.envs.doom.utils import register_custom_components
from sample_factory.envs.doom.doom_params import add_doom_env_args
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.utils.utils import str2bool
from sample_factory.run_algorithm import run_algorithm

def main():
    register_custom_components()
    parser = arg_parser()
    cfg = parse_args(parser=parser)
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
