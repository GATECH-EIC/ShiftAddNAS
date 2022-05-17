# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import torch
import random

import numpy as np
import fairseq.utils as utils

from fairseq import progress_bar
from latency_predictor import LatencyPredictor


class Converter(object):
    def __init__(self, args):
        self.args = args
        self.super_encoder_layer_num = args.encoder_layers
        self.super_decoder_layer_num = args.decoder_layers

        self.encoder_embed_choice = args.encoder_embed_choice
        self.decoder_embed_choice = args.decoder_embed_choice

        self.encoder_ffn_embed_dim_choice = args.encoder_ffn_embed_dim_choice
        self.decoder_ffn_embed_dim_choice = args.decoder_ffn_embed_dim_choice

        self.encoder_layer_num_choice = args.encoder_layer_num_choice
        self.decoder_layer_num_choice = args.decoder_layer_num_choice

        self.encoder_self_attention_heads_choice = args.encoder_self_attention_heads_choice
        self.decoder_self_attention_heads_choice = args.decoder_self_attention_heads_choice
        self.decoder_ende_attention_heads_choice = args.decoder_ende_attention_heads_choice

        self.decoder_arbitrary_ende_attn_choice = args.decoder_arbitrary_ende_attn_choice

        self.encoder_block_types_choice = args.encoder_block_types
        self.decoder_block_types_choice = args.decoder_block_types


    def config2gene(self, config):
        gene = []

        sample_encoder_layer_num = config['encoder']['encoder_layer_num']

        gene.append(config['encoder']['encoder_embed_dim'])
        gene.append(sample_encoder_layer_num)

        for i in range(self.super_encoder_layer_num):
            if i < sample_encoder_layer_num:
                gene.append(config['encoder']['encoder_ffn_embed_dim'][i])
            else:
                gene.append(config['encoder']['encoder_ffn_embed_dim'][0])

        for i in range(self.super_encoder_layer_num):
            if i < sample_encoder_layer_num:
                gene.append(config['encoder']['encoder_self_attention_heads'][i])
            else:
                gene.append(config['encoder']['encoder_self_attention_heads'][0])

        # new added
        for i in range(self.super_encoder_layer_num):
            if i < sample_encoder_layer_num:
                gene.append(config['encoder']['encoder_block_types'][i])
            else:
                gene.append(config['encoder']['encoder_block_types'][0])



        sample_decoder_layer_num = config['decoder']['decoder_layer_num']

        gene.append(config['decoder']['decoder_embed_dim'])
        gene.append(sample_decoder_layer_num)

        for i in range(self.super_decoder_layer_num):
            if i < sample_decoder_layer_num:
                gene.append(config['decoder']['decoder_ffn_embed_dim'][i])
            else:
                gene.append(config['decoder']['decoder_ffn_embed_dim'][0])

        for i in range(self.super_decoder_layer_num):
            if i < sample_decoder_layer_num:
                gene.append(config['decoder']['decoder_self_attention_heads'][i])
            else:
                gene.append(config['decoder']['decoder_self_attention_heads'][0])

        for i in range(self.super_decoder_layer_num):
            if i < sample_decoder_layer_num:
                gene.append(config['decoder']['decoder_ende_attention_heads'][i])
            else:
                gene.append(config['decoder']['decoder_ende_attention_heads'][0])


        for i in range(self.super_decoder_layer_num):
            gene.append(config['decoder']['decoder_arbitrary_ende_attn'][i])

        # new added
        for i in range(self.super_decoder_layer_num):
            if i < sample_decoder_layer_num:
                gene.append(config['decoder']['decoder_block_types'][i])
            else:
                gene.append(config['decoder']['decoder_block_types'][0])

        return gene

    def gene2config(self, gene):

        config = {
            'encoder': {
                'encoder_embed_dim': None,
                'encoder_layer_num': None,
                'encoder_ffn_embed_dim': None,
                'encoder_self_attention_heads': None,
                'encoder_block_types': None,
            },
            'decoder': {
                'decoder_embed_dim': None,
                'decoder_layer_num': None,
                'decoder_ffn_embed_dim': None,
                'decoder_self_attention_heads': None,
                'decoder_ende_attention_heads': None,
                'decoder_arbitrary_ende_attn': None,
                'decoder_block_types': None,
            }
        }
        current_index = 0


        config['encoder']['encoder_embed_dim'] = gene[current_index]
        current_index += 1

        config['encoder']['encoder_layer_num'] = gene[current_index]
        current_index += 1

        config['encoder']['encoder_ffn_embed_dim'] = gene[current_index: current_index + self.super_encoder_layer_num]
        current_index += self.super_encoder_layer_num

        config['encoder']['encoder_self_attention_heads'] = gene[current_index: current_index + self.super_encoder_layer_num]
        current_index += self.super_encoder_layer_num

        # new added
        config['encoder']['encoder_block_types'] = gene[current_index: current_index + self.super_encoder_layer_num]
        current_index += self.super_encoder_layer_num


        config['decoder']['decoder_embed_dim'] = gene[current_index]
        current_index += 1

        config['decoder']['decoder_layer_num'] = gene[current_index]
        current_index += 1

        config['decoder']['decoder_ffn_embed_dim'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        config['decoder']['decoder_self_attention_heads'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        config['decoder']['decoder_ende_attention_heads'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        config['decoder']['decoder_arbitrary_ende_attn'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        # new added
        config['decoder']['decoder_block_types'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num


        return config


    def get_gene_choice(self):
        gene_choice = []

        gene_choice.append(self.encoder_embed_choice)
        gene_choice.append(self.encoder_layer_num_choice)

        for i in range(self.super_encoder_layer_num):
            gene_choice.append(self.encoder_ffn_embed_dim_choice)

        for i in range(self.super_encoder_layer_num):
            gene_choice.append(self.encoder_self_attention_heads_choice)

        # new added
        for i in range(self.super_encoder_layer_num):
            gene_choice.append(self.encoder_block_types_choice)


        gene_choice.append(self.decoder_embed_choice)
        gene_choice.append(self.decoder_layer_num_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_ffn_embed_dim_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_self_attention_heads_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_ende_attention_heads_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_arbitrary_ende_attn_choice)

        # new added
        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_block_types_choice)


        return gene_choice



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Evolution(object):
    def __init__(self, args, trainer, task, epoch_iter):
        self.population_size = args.population_size
        self.args = args
        self.parent_size = args.parent_size
        self.mutation_size = args.mutation_size
        self.mutation_prob = args.mutation_prob
        self.crossover_size = args.crossover_size
        assert self.population_size == self.parent_size + self.mutation_size + self.crossover_size
        self.converter = Converter(args)
        self.gene_choice = self.converter.get_gene_choice()
        self.gene_len = len(self.gene_choice)
        self.evo_iter = args.evo_iter
        self.trainer = trainer
        self.task=task
        self.epoch_iter = epoch_iter
        self.latency_predictor = LatencyPredictor(
            feature_norm=args.feature_norm,
            feature_dim=args.feature_dim,
            new_feature_dim=args.new_feature_dim,
            lat_norm=args.lat_norm,
            ckpt_path=args.ckpt_path
        )
        self.latency_predictor.load_ckpt()
        self.latency_constraint = args.latency_constraint
        self.flops_constraint = args.flops_constraint

        self.best_config = None

        try:
            self.number_datapoints = args.number_datapoints
        except:
            self.number_datapoints = None

    def generate_loss_latency(self):
        popu = self.random_sample(self.number_datapoints)

        print(f"| Start:")
        scores, losses, flops, latency = self.get_scores(popu)
        popu_scores = losses
        print(f"| Lowest loss: {min(popu_scores)}")
        print(f"| Highest loss: {max(popu_scores)}")
        sorted_ind = np.array(popu_scores).argsort()[:self.parent_size]
        print(f"| Predicted latency for lowest loss model: {self.latency_predictor.predict_lat(self.converter.gene2config(popu[sorted_ind[0]]))}")

        latency = []
        for i in range(self.number_datapoints):
            latency.append(self.latency_predictor.predict_lat(self.converter.gene2config(popu[i])))

        configs = []
        for gene in popu:
            configs.append(self.converter.gene2config(gene))

        return np.array(popu_scores), np.array(latency), np.array(configs)


    def run_evo_search(self):
        popu = self.random_sample(self.population_size)

        # popu_scores, losses, flops, latency = self.get_scores(popu)

        all_scores_list = []

        for i in range(self.evo_iter):
            print(f"| Start Iteration {i}:")
            popu_scores, losses, flops, latency = self.get_scores(popu)
            ind = popu_scores.index(min(popu_scores))
            print(f"| Iteration {i}, Lowest obj: {min(popu_scores)}, Loss: {losses[ind]}, Flops: {flops[ind]}, Latency: {latency[ind]}")

            sorted_ind = np.array(popu_scores).argsort()[:self.parent_size]

            self.best_config = self.converter.gene2config(popu[sorted_ind[0]])
            print(f"| Config for lowest loss model: {self.best_config}")
            print(f"| Predicted latency for lowest loss model: {self.latency_predictor.predict_lat(self.converter.gene2config(popu[sorted_ind[0]]))}")

            parents_popu = [popu[m] for m in sorted_ind]

            parents_score = [popu_scores[m] for m in sorted_ind]
            all_scores_list.append(parents_score)

            mutate_popu = []

            k = 0
            while k < self.mutation_size:
                mutated_gene = self.mutate(random.choices(parents_popu)[0])
                if self.satisfy_constraints(mutated_gene):
                    mutate_popu.append(mutated_gene)
                    k += 1

            crossover_popu = []

            k = 0
            while k < self.crossover_size:
                crossovered_gene = self.crossover(random.sample(parents_popu, 2))
                if self.satisfy_constraints(crossovered_gene):
                    crossover_popu.append(crossovered_gene)
                    k += 1

            popu = parents_popu + mutate_popu + crossover_popu

        return self.best_config


    def crossover(self, genes):
        crossovered_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossovered_gene.append(genes[0][i])
            else:
                crossovered_gene.append(genes[1][i])

        return crossovered_gene


    def mutate(self, gene):
        mutated_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choices(self.gene_choice[i])[0])
            else:
                mutated_gene.append(gene[i])

        return mutated_gene


    def get_scores(self, genes):
        configs = []
        for gene in genes:
            configs.append(self.converter.gene2config(gene))

        scores, losses, flops, latency = validate_all(self.args, self.trainer, self.task, self.epoch_iter, configs, self.latency_predictor)

        return scores, losses, flops, latency

    def satisfy_constraints(self, gene):
        satisfy = True

        config = self.converter.gene2config(gene)

        # print(config)

        if self.latency_predictor.predict_lat(config) > self.latency_constraint:
            satisfy = False

        if cal_flops(config, self.trainer, self.args, self.task) > self.flops_constraint:
            satisfy = False

        return satisfy


    def random_sample(self, sample_num):
        popu = []
        i = 0
        while i < sample_num:
            samp_gene = []
            for k in range(self.gene_len):
                samp_gene.append(random.choices(self.gene_choice[k])[0])

            if self.satisfy_constraints(samp_gene):
                popu.append(samp_gene)
                i += 1

        return popu



def test():
    config = {
        'encoder': {
            'encoder_embed_dim': 512,
            'encoder_layer_num': 4,
            'encoder_ffn_embed_dim': [1024, 1025, 1026, 1027],
            'encoder_self_attention_heads': [4, 5, 6, 7],
        },
        'decoder': {
            'decoder_embed_dim': 512,
            'decoder_layer_num': 5,
            'decoder_ffn_embed_dim': [2048, 2049, 2050, 2051, 2052],
            'decoder_self_attention_heads': [4, 6, 7, 8, 9],
            'decoder_ende_attention_heads': [3, 4, 5, 6, 7],
            'decoder_arbitrary_ende_attn': [1, 2, 3, 4, 5, 6, 7]
        }
    }

    args = Namespace(encoder_layers=6,
                     decoder_layers=7,
                     encoder_embed_choice=[768, 512],
                     decoder_embed_choice=[768, 512],
                     encoder_ffn_embed_dim_choice=[3072, 2048],
                     decoder_ffn_embed_dim_choice=[3072, 2048],
                     encoder_layer_num_choice=[6, 5],
                     decoder_layer_num_choice=[6, 5, 4, 3],
                     encoder_self_attention_heads_choice=[8, 4],
                     decoder_self_attention_heads_choice=[8, 4],
                     decoder_ende_attention_heads_choice=[8],
                     decoder_arbitrary_ende_attn_choice=[1, 2]
                     )



    converter = Converter(args)
    gene_get = converter.config2gene(config)

    print(gene_get)
    print(len(gene_get))

    config_get = converter.gene2config(gene_get)

    print(config_get)

    print(len(converter.get_gene_choice()))
    print(converter.get_gene_choice())

def cal_flops(config, trainer, args, task):
    config = my_parser_config(config)
    trainer.set_sample_config(config)
    trainer._model.profile(mode=True)

    import torchprofile
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['iwslt']
    elif 'wmt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['wmt']
    else:
        raise NotImplementedError
    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]
    macs = torchprofile.profile_macs(trainer._model.cuda(), args=(torch.tensor([-1]).cuda(),torch.tensor([-1]).cuda(),
        torch.tensor([dummy_src_tokens], dtype=torch.long).cuda(), torch.tensor([30]).cuda(), torch.tensor([dummy_prev], dtype=torch.long).cuda()))
    trainer._model.profile(mode=False)
    last_layer_macs = config['decoder']['decoder_embed_dim'] * dummy_sentence_length * len(task.tgt_dict)
    flops = (macs - last_layer_macs) * 2 / 1e10
    return flops

def get_coeff(type):
    print(type)
    if 'lightweight' in type and 'attention' not in type:
        coeff = 0.6
        coeff_ffn = 0.5
    elif 'attention' in type and 'lightweight' not in type:
        coeff = 1.0
        coeff_ffn = 0.5
    elif 'attention' in type and 'lightweight' in type:
        coeff = 1.0
        coeff_ffn = 0.5
    return coeff, coeff_ffn

def cal_coeff_flops(config, token_length):
    encoder_attn_flops = 0
    encoder_ffn_flops = 0
    for i in range(config['encoder']['encoder_layer_num']):
        encoder_attn_flops += 4 * token_length * config['encoder']['encoder_embed_dim'] * 512 + 2 * token_length**2 * 512
        encoder_ffn_flops += token_length * config['encoder']['encoder_embed_dim'] * config['encoder']['encoder_ffn_embed_dim'][i] * 2

        coeff, coeff_ffn = get_coeff(config['encoder']['encoder_block_types'][i][0])
        encoder_attn_flops *= coeff
        encoder_ffn_flops *= coeff_ffn

    # encoder_flops = encoder_attn_flops + encoder_ffn_flops
    encoder_flops = encoder_attn_flops # - encoder_ffn_flops

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

        coeff, coeff_ffn = get_coeff(config['decoder']['decoder_block_types'][i])
        decoder_attn_flops *= coeff
        decoder_ffn_flops *= coeff_ffn

    # decoder_flops = decoder_attn_flops + decoder_ffn_flops
    decoder_flops = decoder_attn_flops - decoder_ffn_flops

    return encoder_flops + decoder_flops

def validate_all(args, trainer, task, epoch_itr, configs, predictor):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
        # Initialize data iterator
    def get_itr():
        itr = task.get_batch_iterator(
            dataset=task.dataset('valid'),
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
            prefix='valid on \'{}\' subset'.format('valid'),
        )
        return progress

    # loss
    for config in configs:
        config = my_parser_config(config)
        # print(config)
        trainer.set_sample_config(config)
        progress = get_itr()

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        valid_cnt = 0
        for sample in progress:
            valid_cnt += 1
            if valid_cnt > args.valid_cnt_max:
                break
            trainer.valid_step(sample)

        valid_losses.append(trainer.get_meter('valid_loss').avg)

    # flops
    total_flops = []
    for config in configs:
        # without coeff:
        """
        config = my_parser_config(config)
        trainer.set_sample_config(config)
        trainer._model.profile(mode=True)

        import torchprofile
        dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
        if 'iwslt' in args.arch:
            dummy_sentence_length = dummy_sentence_length_dict['iwslt']
        elif 'wmt' in args.arch:
            dummy_sentence_length = dummy_sentence_length_dict['wmt']
        else:
            raise NotImplementedError
        dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
        dummy_prev = [7] * (dummy_sentence_length - 1) + [2]
        # macs = torchprofile.profile_macs(trainer._model.cuda(), args=(torch.tensor([dummy_src_tokens], dtype=torch.long).cuda(), torch.tensor([30]).cuda(), torch.tensor([dummy_prev], dtype=torch.long).cuda()))
        macs = torchprofile.profile_macs(trainer._model.cuda(), args=(torch.tensor([dummy_src_tokens], dtype=torch.long).cuda(), torch.tensor([30]).cuda(), torch.tensor([dummy_prev], dtype=torch.long).cuda()))
        trainer._model.profile(mode=False)
        last_layer_macs = config['decoder']['decoder_embed_dim'] * dummy_sentence_length * len(task.tgt_dict)
        flops = (macs - last_layer_macs) * 2 / 1e10
        print(f"| Total FLOPs without last layer: {flops} \n")
        total_flops.append(flops)
        """

        # with coeff
        config = my_parser_config(config)
        macs = cal_coeff_flops(config, token_length=30)
        flops = 2 * macs / 1e10
        # print(f"| Total FLOPs: {flops} \n")
        total_flops.append(flops)


    # latency
    total_latency = []
    for config in configs:
        config = my_parser_config(config)
        latency = predictor.predict_lat(config) / 100
        # print(f"| Latency : {latency} s\n")
        total_latency.append(latency)

    obj = list(np.array(valid_losses) - np.array(total_flops))
    # obj = list(np.array(valid_losses) - np.array(total_latency) - np.array(total_flops))
    # obj = list(np.array(valid_losses) + np.array(total_latency))

    # return valid_losses
    return obj, valid_losses, total_flops, total_latency

def my_parser_config(config):
    # this is for solving the inconsistency: [[], [], ..] vs. [ , , ...]
    config['encoder']['encoder_block_types'] = [[item] if not isinstance(item, list) else item for item in config['encoder']['encoder_block_types']]
    return config

if __name__=='__main__':
    test()
