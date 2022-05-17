#!/bin/bash
## SBATCH config file
#SBATCH --job-name=hy34_build_modules
#SBATCH --partition=V100
#SBATCH --reservation=ICLR21
#SBATCH –-nodes=1
#SBATCH --ntasks=1
#SBATCH –-cpus-per-task=8 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=asimov-230
#SBATCH –-signal=USR1@600
#SBATCH --output=outputfile_%j.log

## env
echo $CUDA_VISIBLE_DEVICES

source /mnt/home/v_youhaoran/anaconda3/etc/profile.d/conda.sh
source /mnt/home/v_youhaoran/.bashrc
conda activate nas

echo 'start'
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

cd fairseq/modules/lightadd_layer
python setup.py install
cd ../../..
cd fairseq/modules/lightconv_layer
python setup.py install
cd ../../..
cd fairseq/modules/lightshiftadd_layer
python setup.py install
cd ../../..

echo 'end'
