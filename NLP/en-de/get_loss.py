# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import torch
import pdb

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.trainer import Trainer
from fairseq.evolution import Evolution

import numpy as np

def main(args):
    utils.import_user_module(args)
    utils.handle_save_path(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)

    # Build trainer
    trainer = Trainer(args, task, model, criterion)

    # Load the latest checkpoint if one is available and restore the corresponding train iterator
    args.train_subset = 'valid' # no need to train, so just set a small subset to save loading time
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # run evolutionary search to find the model with lowest loss and satisfies the latency requirement
    evolver = Evolution(args, trainer, task, epoch_itr)
    # best_config = evolver.run_evo_search()

    loss, latency, config = evolver.generate_loss_latency()

    np.save('./figures/loss_{}.npy'.format(args.number_datapoints), loss)
    np.save('./figures/latency_{}.npy'.format(args.number_datapoints), latency)
    np.save('./figures/config_{}.npy'.format(args.number_datapoints), config)


def cli_main():
    parser = options.get_training_parser()
    parser.add_argument('--evo-configs', required=True, is_config_file=True)
    parser.add_argument('--evo-iter', type=int, default=30)
    parser.add_argument('--population-size', type=int, default=125)
    parser.add_argument('--parent-size', type=int, default=25)
    parser.add_argument('--mutation-size', type=int, default=50)
    parser.add_argument('--crossover-size', type=int, default=50)
    parser.add_argument('--mutation-prob', type=float, default=0.3)

    parser.add_argument('--feature-norm', type=float, nargs='+', default=[640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2], help='normalizing factor for each feature')
    parser.add_argument('--feature-dim', type=int, default=94)
    parser.add_argument('--new-feature-dim', type=int, default=12, help='dimension of feature vector')

    parser.add_argument('--lat-norm', type=float, default=200, help='normalizing factor for latency')
    parser.add_argument('--ckpt-path', type=str, help='path to load latency predictor weights')

    parser.add_argument('--latency-constraint', type=float, default=-1, help='latency constraint')
    parser.add_argument('--valid-cnt-max', type=int, default=1e9, help='max number of sentences to use in validation set')

    parser.add_argument('--write-config-path', type=str, help='path to write out the searched best SubTransformer')

    parser.add_argument('--number_datapoints', type=int, default=2000, help='number of datapoints')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    args.feature_norm.extend([1]*(args.feature_dim-len(args.feature_norm)))

    if args.pdb:
        pdb.set_trace()

    # one GPU is fast enough to do the search
    args.distributed_world_size = 1

    # if search on CPU, use fp32 as default
    if args.cpu:
        args.fp16 = False

    main(args)


if __name__ == '__main__':
    cli_main()
