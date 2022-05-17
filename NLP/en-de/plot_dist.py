import collections
import math
import random
import torch
import pdb
import os
import numpy as np
import shutil

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from copy import deepcopy

def my_parser_config(config):
    # this is for solving the inconsistency: [[], [], ..] vs. [ , , ...]
    config['encoder']['encoder_block_types'] = [[item] if not isinstance(item, list) else item for item in config['encoder']['encoder_block_types']]
    return config

def main(args, init_distributed=False):
    utils.import_user_module(args)
    utils.handle_save_path(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(f"| Configs: {args}")

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(f"| Model: {args.arch} \n| Criterion: {criterion.__class__.__name__}")

    # Log architecture
    if args.train_subtransformer:
        print(" \n\n\t\tWARNING!!! Training one single SubTransformer\n\n")
        print(f"| SubTransformer Arch: {utils.get_subtransformer_config(args)} \n")
    else:
        print(" \n\n\t\tWARNING!!! Training SuperTransformer\n\n")
        print(f"| SuperTransformer Arch: {model} \n")

    # Log model size
    if args.train_subtransformer:
        print(f"| SubTransformer size (without embedding weights): {model.get_sampled_params_numel(my_parser_config(utils.get_subtransformer_config(args)))}")
        embed_size = args.decoder_embed_dim_subtransformer * len(task.tgt_dict)
        print(f"| Embedding layer size: {embed_size} \n")

    else:
        model_s = 0
        # if use model.state_dict, then will add 2 more parameters, they are encoder.version and decoder.version. Should not count them
        for name, param in model.named_parameters():
            if 'embed' not in name:
                model_s += param.numel()
        print(f"| SuperTransofmer model size (without embedding weights): {model_s}")

        print(f"| Embedding layer size: {sum(p.numel() for p in model.parameters() if p.requires_grad) - model_s} \n")


    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['iwslt']
    elif 'wmt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['wmt']
    else:
        raise NotImplementedError

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    # profile the overall FLOPs number
    import torchprofile
    config_subtransformer = utils.get_subtransformer_config(args)
    config_subtransformer = my_parser_config(config_subtransformer)

    model.set_sample_config(config_subtransformer)
    model.profile(mode=True)

    macs = torchprofile.profile_macs(model.cuda(), args=(torch.tensor([-1]).cuda(),torch.tensor([-1]).cuda(),
        torch.tensor([dummy_src_tokens], dtype=torch.long).cuda(), torch.tensor([30]).cuda(), torch.tensor([dummy_prev], dtype=torch.long).cuda()))
    model.profile(mode=False)

    last_layer_macs = config_subtransformer['decoder']['decoder_embed_dim'] * dummy_sentence_length * len(task.tgt_dict)

    print(f"| Total FLOPs: {macs * 2}")
    print(f"| Last layer FLOPs: {last_layer_macs * 2}")
    print(f"| Total FLOPs without last layer: {(macs - last_layer_macs) * 2} \n")


    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print(f"| Training on {args.distributed_world_size} GPUs")
    print(f"| Max tokens per GPU = {args.max_tokens} and max sentences per GPU = {args.max_sentences} \n")

    # Load the latest checkpoint if one is available and restore the corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Evaluate the SubTransformer
    config = utils.get_subtransformer_config(args)
    config = my_parser_config(config)
    # print(config)
    trainer.set_sample_config(config)
    valid_loss = validate(args, trainer, task, epoch_itr, ['valid'], 'SubTransformer')
    print(f"| SubTransformer validation loss:{valid_loss}")


    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(9,2.5))


    # for name, param in model.named_parameters():
    #     # print(name)
    #     if name == 'encoder.layers.0.self_attn_lightweight_conv.branches.1.weight':
    #         weight_0 = param.data.reshape(-1)
    #         weight_0 = weight_0[weight_0.nonzero()]

    #     if name == 'encoder.layers.1.self_attn_lightweight_conv.branches.1.weight':
    #         weight_1 = param.data.reshape(-1)
    #         weight_1 = weight_1[weight_1.nonzero()]

    #     if name == 'encoder.layers.2.self_attn_lightweight_conv.branches.1.weight':
    #         weight_2 = param.data.reshape(-1)
    #         weight_2 = weight_2[weight_2.nonzero()]

    #     if name == 'encoder.layers.3.self_attn_lightweight_add.branches.1.weight':
    #         add_0 = param.data.reshape(-1)
    #         add_0 = add_0[add_0.nonzero()]

    #     if name == 'encoder.layers.4.self_attn_lightweight_add.branches.1.weight':
    #         add_1 = param.data.reshape(-1)
    #         add_1 = add_1[add_1.nonzero()]

    #     if name == 'decoder.layers.3.self_attn_lightweight_conv.branches.1.weight':
    #         weight_3 = param.data.reshape(-1)
    #         weight_3 = weight_3[weight_3.nonzero()]

    # weight = torch.cat((weight_0, weight_1), dim=0)
    # weight = torch.cat((weight, weight_2), dim=0)
    # weight = torch.cat((weight, weight_3), dim=0)

    # add = torch.cat((add_0, add_1), dim=0)

    for name, param in model.named_parameters():
        # print(name)
        if name == 'encoder.layers.0.self_attn_lightweight_add.branches.1.weight':
            weight_0 = param.data.reshape(-1)
            weight_0 = weight_0[weight_0.nonzero()]

        if name == 'encoder.layers.1.self_attn_lightweight_shiftadd.branches.1.weight':
            weight_1 = param.data.reshape(-1)
            weight_1 = weight_1[weight_1.nonzero()]

        if name == 'encoder.layers.2.self_attn_lightweight_add.branches.1.weight':
            weight_2 = param.data.reshape(-1)
            weight_2 = weight_2[weight_2.nonzero()]

        if name == 'encoder.layers.3.self_attn_lightweight_add.branches.1.weight':
            weight_3 = param.data.reshape(-1)
            weight_3 = weight_3[weight_3.nonzero()]

        if name == 'encoder.layers.4.self_attn_lightweight_add.branches.1.weight':
            weight_4 = param.data.reshape(-1)
            weight_4 = weight_4[weight_4.nonzero()]

        if name == 'encoder.layers.5.self_attn_lightweight_add.branches.1.weight':
            weight_5 = param.data.reshape(-1)
            weight_5 = weight_5[weight_5.nonzero()]

    weight = torch.cat((weight_0, weight_1), dim=0)
    weight = torch.cat((weight, weight_2), dim=0)
    weight = torch.cat((weight, weight_3), dim=0)
    weight = torch.cat((weight, weight_4), dim=0)
    weight = torch.cat((weight, weight_5), dim=0)

    plot_fig_weights(weight.reshape(1, -1), fig, 1)
    # plot_fig_weights(add.reshape(1, -1), fig, 2)

    plt.tight_layout()
    plt.savefig('./figures/weight_vis.svg')



# plots histogram of weights
def plot_fig_weights(flat_weights, fig, title_name):
    NUM_PLOT_BINS = 10
    font_board = 2

    ax = fig.add_subplot(1, 4, title_name)
    # ax.set_title("Pruning " + save_name[title_name-1] + '%')
    ax.set_title("")
    ax.hist(flat_weights.cpu(), NUM_PLOT_BINS, color='green', alpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_linewidth(font_board)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(font_board)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_linewidth(font_board)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_linewidth(font_board)
    ax.spines['right'].set_color('black')


def validate(args, trainer, task, epoch_itr, subsets, sampled_arch_name):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        def get_itr():
            itr = task.get_batch_iterator(
                dataset=task.dataset(subset),
                max_tokens=args.max_tokens_valid,
                max_sentences=args.max_sentences_valid,
                max_positions=utils.resolve_max_positions(
                    task.max_positions(),
                    trainer.get_model().max_positions(),
                ),
                ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=args.required_batch_size_multiple,
                seed=args.seed,
                num_shards=args.distributed_world_size,
                shard_id=args.distributed_rank,
                num_workers=args.num_workers,
            ).next_epoch_itr(shuffle=False)
            progress = progress_bar.build_progress_bar(
                args, itr, epoch_itr.epoch,
                prefix='validate on \'{}\' subset'.format(subset),
            )
            return progress
        progress = get_itr()

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample, num_bits=args.num_bits, num_bits_grad=args.num_bits_grad)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = utils.get_valid_stats(trainer, args)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg

        # log validation stats
        stats = utils.get_valid_stats(trainer, args, extra_meters)

        stats[sampled_arch_name+'_loss'] = deepcopy(stats['loss'])
        stats[sampled_arch_name+'_nll_loss'] = deepcopy(stats['nll_loss'])

        for k, meter in extra_meters.items():
            stats[k] = meter.avg

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses



def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)

def cli_main():
    parser = options.get_training_parser()
    parser.add_argument('--train-subtransformer', action='store_true', default=False, help='whether train SuperTransformer or SubTransformer')
    parser.add_argument('--sub-configs', required=False, is_config_file=True, help='when training SubTransformer, use --configs to specify architecture and --sub-configs to specify other settings')

    # for profiling
    parser.add_argument('--profile-flops', action='store_true', help='measure the FLOPs of a SubTransformer')

    parser.add_argument('--latgpu', action='store_true', help='measure SubTransformer latency on GPU')
    parser.add_argument('--latcpu', action='store_true', help='measure SubTransformer latency on CPU')
    parser.add_argument('--latiter', type=int, default=300, help='how many iterations to run when measure the latency')
    parser.add_argument('--latsilent', action='store_true', help='keep silent when measure latency')

    parser.add_argument('--validate-subtransformer', action='store_true', help='evaluate the SubTransformer on the validation set')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    if args.latcpu:
        args.cpu = True
        args.fp16 = False

    if args.latgpu or args.latcpu or args.profile_flops:
        args.distributed_world_size = 1

    if args.pdb:
        pdb.set_trace()

    main(args)


if __name__ == '__main__':
    cli_main()
