
best_config_120 = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 1024, 1024, 1024, 3072], 'encoder_self_attention_heads': [8, 4, 8, 8, 16, 16], 'encoder_block_types': ['lightweight_shiftadd', 'self_attention+lightweight_shiftadd', 'self_attention+lightweight_shiftadd', 'self_attention+lightweight_shiftadd', 'self_attention', 'lightweight_add']}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 3, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 16, 16, 8, 4, 16], 'decoder_ende_attention_heads': [4, 4, 4, 16, 16, 16], 'decoder_arbitrary_ende_attn': [-1, 1, 2, -1, 2, -1], 'decoder_block_types': ['self_attention+lightweight_conv', 'self_attention', 'self_attention', 'self_attention+lightweight_shiftadd', 'self_attention', 'self_attention+lightweight_shiftadd']}}
write_config_path_120 = 'configs/wmt14.en-fr/subtransformer/wmt14enfr_gpu_V100_shiftadd_v1@120ms.yml'

best_config_150 = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [2048, 1024, 1024, 3072, 1024, 4096], 'encoder_self_attention_heads': [16, 16, 8, 8, 8, 16], 'encoder_block_types': ['self_attention+lightweight_shiftadd', 'self_attention+lightweight_shiftadd', 'self_attention', 'self_attention+lightweight_conv', 'lightweight_shiftadd', 'lightweight_add']}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 3, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 2048, 3072, 3072], 'decoder_self_attention_heads': [8, 16, 16, 8, 8, 8], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 8], 'decoder_arbitrary_ende_attn': [2, 2, 2, 2, 2, 1], 'decoder_block_types': ['self_attention+lightweight_conv', 'self_attention', 'self_attention', 'self_attention+lightweight_add', 'self_attention', 'self_attention+lightweight_shiftadd']}}
write_config_path_150 = 'configs/wmt14.en-fr/subtransformer/wmt14enfr_gpu_V100_shiftadd_v1@150ms.yml'

best_config_180 = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [2048, 3072, 2048, 3072, 1024, 4096], 'encoder_self_attention_heads': [8, 16, 8, 8, 16, 8], 'encoder_block_types': ['self_attention+lightweight_conv', 'self_attention+lightweight_shiftadd', 'self_attention+lightweight_conv', 'self_attention+lightweight_shiftadd', 'self_attention', 'lightweight_add']}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 4, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 1024, 3072], 'decoder_self_attention_heads': [8, 16, 16, 4, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 16, 16], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 1, 1], 'decoder_block_types': ['self_attention+lightweight_conv', 'self_attention', 'self_attention', 'self_attention', 'self_attention+lightweight_conv', 'self_attention+lightweight_shiftadd']}}
write_config_path_180 = 'configs/wmt14.en-fr/subtransformer/wmt14enfr_gpu_V100_shiftadd_v1@180ms.yml'

best_config_200 = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [2048, 4096, 2048, 1024, 4096, 4096], 'encoder_self_attention_heads': [4, 16, 4, 16, 8, 16], 'encoder_block_types': ['self_attention+lightweight_add', 'self_attention+lightweight_add', 'self_attention', 'self_attention+lightweight_conv', 'self_attention+lightweight_conv', 'lightweight_add']}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 4, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 2048, 2048, 3072], 'decoder_self_attention_heads': [8, 16, 16, 4, 16, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 8, 8], 'decoder_arbitrary_ende_attn': [2, 2, 2, 2, -1, 1], 'decoder_block_types': ['self_attention+lightweight_conv', 'self_attention', 'self_attention', 'self_attention', 'self_attention', 'self_attention+lightweight_shiftadd']}}
write_config_path_200 = 'configs/wmt14.en-fr/subtransformer/wmt14enfr_gpu_V100_shiftadd_v1@200ms.yml'

best_config_250 = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [4096, 1024, 2048, 3072, 1024, 4096], 'encoder_self_attention_heads': [4, 4, 8, 16, 4, 8], 'encoder_block_types': ['self_attention', 'lightweight_conv', 'self_attention+lightweight_shiftadd', 'self_attention+lightweight_shiftadd', 'self_attention', 'lightweight_add']}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 5, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 2048, 2048, 2048], 'decoder_self_attention_heads': [8, 16, 16, 16, 4, 8], 'decoder_ende_attention_heads': [4, 4, 4, 8, 8, 8], 'decoder_arbitrary_ende_attn': [2, 2, 2, 2, 2, 2], 'decoder_block_types': ['self_attention+lightweight_conv', 'self_attention', 'self_attention', 'self_attention', 'self_attention', 'self_attention+lightweight_shiftadd']}}
write_config_path_250 = 'configs/wmt14.en-fr/subtransformer/wmt14enfr_gpu_V100_shiftadd_v1@250ms.yml'

best_config_300 = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 2048, 4096, 1024, 3072, 4096], 'encoder_self_attention_heads': [8, 16, 16, 16, 16, 8], 'encoder_block_types': ['lightweight_add', 'self_attention+lightweight_conv', 'self_attention', 'self_attention+lightweight_add', 'self_attention', 'lightweight_add']}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 5, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 2048, 1024], 'decoder_self_attention_heads': [8, 8, 16, 16, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 4, 16, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, -1], 'decoder_block_types': ['self_attention+lightweight_conv', 'self_attention', 'self_attention', 'self_attention', 'self_attention', 'self_attention']}}
write_config_path_300 = 'configs/wmt14.en-fr/subtransformer/wmt14enfr_gpu_V100_shiftadd_v1@300ms.yml'

best_config_350 = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 2048, 3072, 3072, 1024, 3072], 'encoder_self_attention_heads': [16, 8, 4, 8, 16, 8], 'encoder_block_types': ['self_attention+lightweight_conv', 'self_attention+lightweight_add', 'self_attention+lightweight_conv', 'self_attention+lightweight_add', 'self_attention', 'lightweight_add']}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 5, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 16, 16, 16, 4, 16], 'decoder_ende_attention_heads': [4, 4, 4, 4, 16, 16], 'decoder_arbitrary_ende_attn': [2, 2, 2, 2, 2, 2], 'decoder_block_types': ['self_attention+lightweight_conv', 'self_attention', 'self_attention', 'self_attention', 'self_attention', 'self_attention+lightweight_conv']}}
write_config_path_350 = 'configs/wmt14.en-fr/subtransformer/wmt14enfr_gpu_V100_shiftadd_v1@350ms.yml'

def write_config(write_config_path, best_config):
    with open(write_config_path, 'w') as fid:
        # ablation
        best_config['encoder']['encoder_block_types'][-1] = 'self_attention'

        encoder_layer_num = best_config['encoder']['encoder_layer_num']
        decoder_layer_num = best_config['decoder']['decoder_layer_num']

        fid.write(f"encoder-embed-dim-subtransformer: {best_config['encoder']['encoder_embed_dim']}\n")
        fid.write(f"decoder-embed-dim-subtransformer: {best_config['decoder']['decoder_embed_dim']}\n\n")

        fid.write(f"encoder-ffn-embed-dim-all-subtransformer: {best_config['encoder']['encoder_ffn_embed_dim'][:encoder_layer_num]}\n")
        fid.write(f"decoder-ffn-embed-dim-all-subtransformer: {best_config['decoder']['decoder_ffn_embed_dim'][:decoder_layer_num]}\n\n")

        fid.write(f"encoder-layer-num-subtransformer: {best_config['encoder']['encoder_layer_num']}\n")
        fid.write(f"decoder-layer-num-subtransformer: {best_config['decoder']['decoder_layer_num']}\n\n")

        fid.write(f"encoder-self-attention-heads-all-subtransformer: {best_config['encoder']['encoder_self_attention_heads'][:encoder_layer_num]}\n")
        fid.write(f"decoder-self-attention-heads-all-subtransformer: {best_config['decoder']['decoder_self_attention_heads'][:decoder_layer_num]}\n")
        fid.write(f"decoder-ende-attention-heads-all-subtransformer: {best_config['decoder']['decoder_ende_attention_heads'][:decoder_layer_num]}\n\n")

        fid.write(f"decoder-arbitrary-ende-attn-all-subtransformer: {best_config['decoder']['decoder_arbitrary_ende_attn'][:decoder_layer_num]}\n\n")

        # fid.write(f"encoder-block-types-all-subtransformer: {best_config['encoder']['encoder_block_types']}\n")
        # fid.write(f"decoder-block-types-all-subtransformer: {best_config['decoder']['decoder_block_types']}\n")

        fid.write("encoder-block-types-all-subtransformer: [{}]\n".format(', '.join(map(str, best_config['encoder']['encoder_block_types']))))
        fid.write("decoder-block-types-all-subtransformer: [{}]\n".format(', '.join(map(str, best_config['decoder']['decoder_block_types']))))

        fid.close()

if __name__ == '__main__':
    write_config(write_config_path_120, best_config_120)
    write_config(write_config_path_150, best_config_150)
    write_config(write_config_path_180, best_config_180)
    write_config(write_config_path_200, best_config_200)
    write_config(write_config_path_250, best_config_250)
    write_config(write_config_path_300, best_config_300)
    write_config(write_config_path_350, best_config_350)