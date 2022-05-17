# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

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

def cal_flops(config, token_length):
    encoder_attn_flops = 0
    encoder_ffn_flops = 0
    for i in range(config['encoder']['encoder_layer_num']):
        encoder_attn_flops += 4 * token_length * config['encoder']['encoder_embed_dim'] * 512 + 2 * token_length**2 * 512
        encoder_ffn_flops += token_length * config['encoder']['encoder_embed_dim'] * config['encoder']['encoder_ffn_embed_dim'][i] * 2
    encoder_flops = encoder_attn_flops + encoder_ffn_flops

    decoder_attn_flops = 0
    decoder_ffn_flops = 0
    for i in range(config['decoder']['decoder_layer_num']):
        if config['decoder']['decoder_arbitrary_ende_attn'][i] == -1:
            decoder_attn_flops += (4 * token_length * config['decoder']['decoder_embed_dim'] * 512 + 2 * token_length**2 * 512) * 2
        elif config['decoder']['decoder_arbitrary_ende_attn'][i] == 1:
            decoder_attn_flops += 4 * token_length * config['decoder']['decoder_embed_dim'] * 512 + 2 * token_length**2 * 512
            decoder_attn_flops += 4 * token_length * config['decoder']['decoder_embed_dim'] * 512 * 1.5 + 2 * token_length**2 * 512 * 2
        elif config['decoder']['decoder_arbitrary_ende_attn'][i] == 2:
            decoder_attn_flops += 4 * token_length * config['decoder']['decoder_embed_dim'] * 512 + 2 * token_length**2 * 512
            decoder_attn_flops += 4 * token_length * config['decoder']['decoder_embed_dim'] * 512 * 2 + 2 * token_length**2 * 512 * 3
        decoder_ffn_flops += token_length * config['decoder']['decoder_embed_dim'] * config['decoder']['decoder_ffn_embed_dim'][i] * 2
    decoder_flops = decoder_attn_flops + decoder_ffn_flops

    return encoder_flops + decoder_flops

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
    if args.profile_flops:
        import torchprofile
        config_subtransformer = utils.get_subtransformer_config(args)
        config_subtransformer = my_parser_config(config_subtransformer)

        config_subtransformer['decoder']['decoder_arbitrary_ende_attn'] = [-1, -1, 2, 2, 1, -1]
        print(config_subtransformer)

        cal_macs = cal_flops(config_subtransformer, dummy_sentence_length)

        model.set_sample_config(config_subtransformer)
        model.profile(mode=True)
        ### macs = torchprofile.profile_macs(model, args=(torch.tensor([dummy_src_tokens], dtype=torch.long), torch.tensor([30]), torch.tensor([dummy_prev], dtype=torch.long)))

        # for k, m in enumerate(model.modules()):
        #     print(k, m)

        macs = torchprofile.profile_macs(model.cuda(), args=(torch.tensor([-1]).cuda(),torch.tensor([-1]).cuda(),
            torch.tensor([dummy_src_tokens], dtype=torch.long).cuda(), torch.tensor([30]).cuda(), torch.tensor([dummy_prev], dtype=torch.long).cuda()))
        model.profile(mode=False)

        last_layer_macs = config_subtransformer['decoder']['decoder_embed_dim'] * dummy_sentence_length * len(task.tgt_dict)

        print(f"| Total FLOPs: {macs * 2}")
        print(f"| Last layer FLOPs: {last_layer_macs * 2}")
        print(f"| Total FLOPs without last layer: {(macs - last_layer_macs) * 2} \n")

        print(f"| Cal. FLOPs: {2 * cal_macs}")
        exit(0)

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print(f"| Training on {args.distributed_world_size} GPUs")
    print(f"| Max tokens per GPU = {args.max_tokens} and max sentences per GPU = {args.max_sentences} \n")

    # Measure model latency, the program will exit after profiling latency
    if args.latcpu or args.latgpu:
        utils.measure_latency(args, model, dummy_src_tokens, dummy_prev)
        exit(0)

    # Load the latest checkpoint if one is available and restore the corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Evaluate the SubTransformer
    if args.validate_subtransformer:
        config = utils.get_subtransformer_config(args)
        config = my_parser_config(config)
        # print(config)
        trainer.set_sample_config(config)
        valid_loss = validate(args, trainer, task, epoch_itr, ['valid'], 'SubTransformer')
        print(f"| SubTransformer validation loss:{valid_loss}")

    # Loop boundaries
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()

    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')

    represent_configs = utils.get_represent_configs(args)

    # Main training loop
    global history_score
    prev_record = os.path.join(args.save_dir, 'record.txt')
    if os.path.exists(prev_record):
        f = open(prev_record, 'r')
        content = f.readlines()
        losses = []
        ppls = []
        bleus = []
        for line in content:
            loss  = float(line.split(',')[0][:-2])
            ppl   = float(line.split(',')[1][:-2])
            bleu  = float(line.split(',')[2][:-2])
            losses.append(loss)
            ppls.append(ppl)
            bleus.append(bleu)
        f.close()
        history_score = np.array([losses, ppls, bleus]).T
    else:
        history_score = np.zeros((150, 3))

    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            for k, v in represent_configs.items():
                print('previous: ', v)
                v = my_parser_config(v)
                print('update: ', v)
                trainer.set_sample_config(config=v)
                valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, sampled_arch_name=k)
        else:
            valid_losses = [None]

        # update the best loss and get current lr; the real lr scheduling is done in trainer.train_step()
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint epoch level
        if epoch_itr.epoch % args.save_interval == 0:
            print('begin saving ...')
            _ = checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
            print('end saving ...')

            txt_name = 'last_bleus.txt'
            f = open(txt_name, 'r')
            bleus = f.readlines()
            bleus_score = float(bleus[0].split(',')[0].split(' ')[-1])
            f.close()

            prev_bleus = max(history_score[:, 2])

            print(bleus_score, prev_bleus)

            if bleus_score > prev_bleus:
                print('saving best ckpt ...')
                last_ckpt = os.path.join(args.save_dir, 'checkpoint_last.pt')
                best_ckpt = os.path.join(args.save_dir, 'checkpoint_best_bleus.pt'.format(bleus_score))
                if os.path.exists(last_ckpt):
                    shutil.copyfile(last_ckpt, best_ckpt)

            history_score[int(epoch_itr.epoch - 1)][2] = bleus_score

            np.savetxt(os.path.join(args.save_dir, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')

    train_meter.stop()
    print('| Done training in {:.1f} seconds'.format(train_meter.sum))


def cyclic_adjust_precision(args, num_update, cyclic_period):
    assert len(args.cyclic_num_bits_schedule) == 2
    assert len(args.cyclic_num_grad_bits_schedule) == 2

    num_bit_min = args.cyclic_num_bits_schedule[0]
    num_bit_max = args.cyclic_num_bits_schedule[1]

    num_grad_bit_min = args.cyclic_num_grad_bits_schedule[0]
    num_grad_bit_max = args.cyclic_num_grad_bits_schedule[1]

    args.num_bits = np.rint(num_bit_min +
                            0.5 * (num_bit_max - num_bit_min) *
                            (1 + np.cos(np.pi * ((num_update % cyclic_period) / cyclic_period) + np.pi)))
    args.num_grad_bits = np.rint(num_grad_bit_min +
                                 0.5 * (num_grad_bit_max - num_grad_bit_min) *
                                 (1 + np.cos(np.pi * ((num_update % cyclic_period) / cyclic_period) + np.pi)))

    if num_update % args.print_precision == 0:
        print('Iter [{}] num_bits = {} num_grad_bits = {} cyclic precision'.format(num_update, args.num_bits,
                                                                                          args.num_grad_bits))

    return args


def train(args, trainer, task, epoch_itr):

    global history_score

    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch,
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf

    represent_configs = utils.get_represent_configs(args)

    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        if args.train_subtransformer:
            # training one SubTransformer only
            configs = [my_parser_config(utils.get_subtransformer_config(args))]
        else:
            # training SuperTransformer by randomly sampling SubTransformers
            configs = [utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=True, rand_seed=trainer.get_num_updates(),
                                            super_decoder_num_layer=args.decoder_layers)]

        # cpt
        num_update = trainer.get_num_updates()
        max_update = args.max_update

        cyclic_period = int(max_update / args.num_cyclic_period)
        args = cyclic_adjust_precision(args, num_update, cyclic_period)

        log_output = trainer.train_step(samples, configs=configs, num_bits=args.num_bits, num_bits_grad=args.num_bits_grad)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = utils.get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg

        utils.log_arch_info(stats, configs[0])

        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            for k, v in represent_configs.items():
                trainer.set_sample_config(config=v)
                valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, sampled_arch_name=k)

            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break



    # log end-of-epoch stats
    stats = utils.get_training_stats(trainer)
    for k, meter in extra_meters.items():
        print(k)
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # print(stats)
    history_score[int(epoch_itr.epoch - 1)][0] = float(stats['loss'].avg)
    history_score[int(epoch_itr.epoch - 1)][1] = float(stats['ppl'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


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

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
