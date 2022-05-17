#!/bin/bash
## SBATCH config file
#SBATCH --job-name=hy34_train_subtransformer_2
#SBATCH --partition=2080Ti_mlong
#SBATCH –-nodes=1
#SBATCH --ntasks=1
#SBATCH –-cpus-per-task=40 
#SBATCH --gres=gpu:8
#SBATCH --nodelist=asimov-204
#SBATCH –-signal=USR1@600
#SBATCH --time=24:00:00
#SBATCH --output=outputfile_%j.log

## env
source /mnt/home/qiuling/anaconda3/etc/profile.d/conda.sh
source /mnt/home/qiuling/.bashrc
export CUDA_HOME=/mnt/archive/qiuling/cuda-10.2
export PATH=$PATH:/mnt/archive/qiuling/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/archive/qiuling/cuda-10.2/lib64
conda activate nas

echo 'start'
echo $CUDA_VISIBLE_DEVICES
nvidia-smi


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--configs=configs/wmt14.en-de/subtransformer/sample_test_2.yml \
--sub-configs=configs/wmt14.en-de/subtransformer/common_shiftadd.yml

echo 'end'
