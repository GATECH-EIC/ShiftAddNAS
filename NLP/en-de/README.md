# ShiftAddNAS on NLP Tasks

## Usage

### Installation
To install from source and develop locally:

```bash
cd NLP/en-de/
pip install --editable .

# build all required operators (lightadd/lightconv/lightshiftadd)
cd fairseq/modules/lightadd_layer (or conv or shiftadd)
python setup.py install

export PYTHONPATH=local_path/fairseq/modules/lightadd_layer
export PYTHONPATH=$PYTHONPATH:local_path/fairseq/modules/lightconv_layer
export PYTHONPATH=$PYTHONPATH:local_path/fairseq/modules/lightshiftadd_layer
echo $PYTHONPATH
```

### Data Preparation

| Task | task_name | Train | Valid | Test | 
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| WMT'14 En-De | wmt14.en-de | [WMT'16](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | newstest2013 | newstest2014 | 
| WMT'14 En-Fr | wmt14.en-fr | [WMT'14](http://statmt.org/wmt14/translation-task.html#Download) | newstest2012&2013 | newstest2014 | 

To download and preprocess data, run:
```bash
bash configs/[task_name]/preprocess.sh
```

If you find preprocessing time-consuming, you can directly download the preprocessed data HAT provides:
```bash
bash configs/[task_name]/get_preprocessed.sh
```


### Train and Test

If you want to train or test the searched architectures, refer to commands below:

> Train

````bash
# train ShiftAddNAS

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train.py \
--configs=configs/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@300ms_update.yml \
--sub-configs=configs/wmt14.en-de/subtransformer/common_shiftadd_v3_quant.yml \
--num_bits -1 \
--num_bits_grad -1 \
--update-freq 19

# train ShiftAddNAS (8 bits)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 nohup python train.py \
--configs=configs/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@200ms_Jun.29_quant.yml \
--sub-configs=configs/wmt14.en-de/subtransformer/common_shiftadd_v3.yml \
--num_bits 8 \
--num_bits_grad 8 \
> shiftadd_v3@200ms_quant.out 2>&1 &
````

> Test

````bash
# test ShiftAddNAS

bash configs/[task_name]/test.sh \
    [model_file] \
    configs/[task_name]/subtransformer/[model_name].yml \
    [normal|sacre]

# for example

bash configs/wmt14.en-de/test.sh \
./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@300ms_update/checkpoint_last.pt \
configs/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@300ms_update.yml \
normal \
0 \
test
````


#### Test model size and FLOPs

To profile the model size and FLOPs (FLOPs profiling needs [torchprofile](https://github.com/mit-han-lab/torchprofile.git)), you can run the commands below. By default, only the model size is profiled:

```bash
python train.py \
    --configs=configs/[task_name]/subtransformer/[model_name].yml \
    --sub-configs=configs/[task_name]/subtransformer/common.yml \
    --profile-flops

# for example
CUDA_VISIBLE_DEVICES=0 python train.py \
--configs=configs/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@350ms_Jul.13.yml \
--sub-configs=configs/wmt14.en-de/subtransformer/common_shiftadd_v3_quant.yml \
--profile-flops
```


### Search

#### 1. Train a SuperTransformer

The SuperTransformer is a supernet that contains many SubTransformers with weight-sharing.
By default, we train WMT tasks on 8 GPUs. Please adjust `--update-freq` according to GPU numbers (`128/x` for x GPUs). Note that for IWSLT, we only train on one GPU with `--update-freq=1`. 

```bash
python train.py --configs=configs/[task_name]/supertransformer/[search_space].yml
# for example
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--configs=configs/wmt14.en-de/supertransformer/space_shiftadd_v3_act_1.yml \
--update-freq=16
```
In the `--configs` file, SuperTransformer model architecture, SubTransformer search space and training settings are specified.

We also provide pre-trained SuperTransformers [here]()!


#### 2. Evolutionary Search
The second step of HAT is to perform an evolutionary search in the trained SuperTransformer with a hardware latency constraint in the loop. We train a latency predictor to get fast and accurate latency feedback.

##### 2.1 Generate a latency dataset
```bash
python latency_dataset.py --configs=configs/[task_name]/latency_dataset/[hardware_name].yml

# for example
CUDA_VISIBLE_DEVICES=9 python latency_dataset.py \
--configs=configs/wmt14.en-de/latency_dataset/gpu_2080Ti_shiftadd.yml
```
The `--configs` file contains the design space in which we sample models to get (model_architecture, real_latency) data pairs.

We provide the datasets we collect in the [latency_dataset](./latency_dataset) folder.

##### 2.2 Train a latency predictor
Then train a predictor with collected dataset:
```bash
python latency_predictor.py --configs=configs/[task_name]/latency_predictor/[hardware_name].yml

# for example
CUDA_VISIBLE_DEVICES=8 python latency_predictor.py \
--configs=configs/wmt14.en-de/latency_predictor/gpu_2080Ti_shiftadd.yml
```
The `--configs` file contains the predictor's model architecture and training settings.
We provide pre-trained predictors in [latency_dataset/predictors](./latency_dataset/predictors) folder.

##### 2.3 Run evolutionary search with a latency constraint
```bash
python evo_search.py --configs=[supertransformer_config_file].yml --evo-configs=[evo_settings].yml

# for example
CUDA_VISIBLE_DEVICES=8 python evo_search.py \
--configs=configs/wmt14.en-de/supertransformer/space_shiftadd_v3_act_1.yml \
--evo-configs=configs/wmt14.en-de/evo_search/wmt14ende_2080Ti_shiftadd.yml
```
The `--configs` file points to the SuperTransformer training config file. `--evo-configs` file includes evolutionary search settings, and also specifies the desired latency constraint `latency-constraint`. Note that the `feature-norm` and `lat-norm` here should be the same as those when training the latency predictor. `--write-config-path` specifies the location to write out the searched SubTransformer architecture. 


#### 3. Train a Searched SubTransformer
Finally, we train the search SubTransformer from scratch:
```bash
python train.py --configs=[subtransformer_architecture].yml --sub-configs=configs/[task_name]/subtransformer/common.yml
# for example
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train.py \
--configs=configs/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@300ms_update.yml \
--sub-configs=configs/wmt14.en-de/subtransformer/common_shiftadd_v3_quant.yml \
--num_bits -1 \
--num_bits_grad -1 \
--update-freq 19
```

`--configs` points to the `--write-config-path` in step 2.3. `--sub-configs` contains training settings for the SubTransformer.

After training a SubTransformer, you can test its performance with the methods in [Testing](#testing) section.

### Dependencies
* Python >= 3.6
* [PyTorch](http://pytorch.org/) == 1.4.0
* configargparse >= 0.14
* New model training requires NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)


## Acknowledgements

* We are thankful to [fairseq](https://github.com/pytorch/fairseq) and [HAT](https://github.com/mit-han-lab/hardware-aware-transformers) as the backbone of this repo.