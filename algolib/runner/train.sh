#!/bin/bash
set -x
 
# 0. placeholder
workdir=$(cd $(dirname $1); pwd)
if [[ "$workdir" =~ "submodules/mmseg" ]]
then
    if [ -d "$workdir/algolib/configs" ]
    then
        rm -rf $workdir/algolib/configs
        ln -s $workdir/configs $workdir/algolib/
    else
        ln -s $workdir/configs $workdir/algolib/
    fi
else
    if [ -d "$workdir/submodules/mmseg/algolib/configs" ]
    then
        rm -rf $workdir/submodules/mmseg/algolib/configs
        ln -s $workdir/submodules/mmseg/configs $workdir/submodules/mmseg/algolib/
    else
        ln -s $workdir/submodules/mmseg/configs $workdir/submodules/mmseg/algolib/
    fi
fi
 
# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmseg/$3
export PYTORCH_VERSION=1.4
 
# 2. set time
now=$(date +"%Y%m%d_%H%M%S")
 
# 3. set env
path=$PWD
if [[ "$path" =~ "submodules/mmseg" ]]
then
    pyroot=$path
    comroot=$path/../..
    init_path=$path/..
else
    pyroot=$path/submodules/mmseg
    comroot=$path
    init_path=$path/submodules
fi
echo $pyroot
export PYTHONPATH=$comroot:$pyroot:$PYTHONPATH
export FRAME_NAME=mmseg    #customize for each frame
export MODEL_NAME=$3
 
# mmcv path
CONDA_ROOT=/mnt/cache/share/platform/env/miniconda3.7
MMCV_PATH=${CONDA_ROOT}/envs/${CONDA_DEFAULT_ENV}/mmcvs
mmcv_version=1.3.15
export PYTHONPATH=${MMCV_PATH}/${mmcv_version}:$PYTHONPATH
#export PYTHONPATH=/mnt/lustre/sunxiaoye/Code/mmcv/1.3.15:$PYTHONPATH
#export PYTHONPATH=/mnt/lustre/sunxiaoye/Code/mmcv/pt1.3.15:$PYTHONPATH
export MMCV_HOME=/mnt/lustre/share_data/parrots_algolib/datasets/pretrain/mmcv

# init_path
export PYTHONPATH=$init_path/common/sites/:$PYTHONPATH # necessary for init
 
# 4. build necessary parameter
partition=$1 
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
 
# 5. model choice
export PARROTS_DEFAULT_LOGGER=FALSE


case $MODEL_NAME in
    "fcn_r101-d8_512x512_160k_ade20k")
        FULL_MODEL="fcn/fcn_r101-d8_512x512_160k_ade20k"
        ;;
    "pspnet_r101-d8_512x512_160k_ade20k")
        FULL_MODEL="pspnet/pspnet_r101-d8_512x512_160k_ade20k"
        ;;
    "deeplabv3_r101-d8_512x512_160k_ade20k")
        FULL_MODEL="deeplabv3/deeplabv3_r101-d8_512x512_160k_ade20k"
        ;;
    "deeplabv3plus_r101-d8_512x512_160k_ade20k")
        FULL_MODEL="deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k"
        ;;
    "apcnet_r101-d8_512x512_160k_ade20k")
        FULL_MODEL="apcnet/apcnet_r101-d8_512x512_160k_ade20k"
        ;;
    "fcn_hr18_512x512_160k_ade20k")
        FULL_MODEL="hrnet/fcn_hr18_512x512_160k_ade20k"
        ;;
    "fcn_hr18s_512x512_160k_ade20k")
        FULL_MODEL="hrnet/fcn_hr18s_512x512_160k_ade20k"
        ;;
    "fcn_hr48_512x512_160k_ade20k")
        FULL_MODEL="hrnet/fcn_hr48_512x512_160k_ade20k"
        ;;
    "ocrnet_hr18_512x512_160k_ade20k")
        FULL_MODEL="ocrnet/ocrnet_hr18_512x512_160k_ade20k"
        ;;
    "ocrnet_hr18s_512x512_160k_ade20k")
        FULL_MODEL="ocrnet/ocrnet_hr18s_512x512_160k_ade20k"
        ;;
    "ocrnet_hr48_512x512_160k_ade20k")
        FULL_MODEL="ocrnet/ocrnet_hr48_512x512_160k_ade20k"
        ;;
    "isanet_r101-d8_512x512_160k_ade20k")
        FULL_MODEL="isanet/isanet_r101-d8_512x512_160k_ade20k"
        ;;
    "dmnet_r101-d8_512x512_160k_ade20k")
        FULL_MODEL="dmnet/dmnet_r101-d8_512x512_160k_ade20k"
        ;;
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

set -x

file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

srun -p $1 -n$2 -w SH-IDC1-10-198-4-155\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
