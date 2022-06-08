#!/bin/bash
set -x
set -o pipefail

# 0. check the most important SMART_ROOT
echo  "!!!!!SMART_ROOT is" $SMART_ROOT
if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

# 1. set env_path and build soft links for mm configs
if [[ $PWD =~ "mmseg" ]]
then 
    pyroot=$PWD
else
    pyroot=$PWD/mmseg
fi
echo $pyroot
if [ -d "$pyroot/algolib/configs" ]
then
    rm -rf $pyroot/algolib/configs
    ln -s $pyroot/configs $pyroot/algolib/
else
    ln -s $pyroot/configs $pyroot/algolib/
fi

# 2. build file folder for save log and set time
mkdir -p algolib_gen/mmseg/$3
now=$(date +"%Y%m%d_%H%M%S")
 
# 3. set env variables
export PYTORCH_VERSION=1.4
export PYTHONPATH=$pyroot:$PYTHONPATH
export MODEL_NAME=$3
export FRAME_NAME=mmseg    #customize for each frame
export PARROTS_DEFAULT_LOGGER=FALSE

# 4. init_path
export PYTHONPATH=${SMART_ROOT}:$PYTHONPATH
export PYTHONPATH=${SMART_ROOT}/common/sites:$PYTHONPATH
 
# 5. build necessary parameter
partition=$1  
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
EXTRA_ARGS=${array[@]:3}
EXTRA_ARGS=${EXTRA_ARGS//--resume/--resume-from}
SRUN_ARGS=${SRUN_ARGS:-""}

# 6. model list
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
    # 注：以下三个模型存在相同的问题，详见：https://jira.sensetime.com/browse/PARROTSXQ-7694?filter=-2
    # "upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K")
    #     FULL_MODEL="swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K"
    #     ;;
    # "upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K")
    #     FULL_MODEL="swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K"
    #     ;;
    # "upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K")
    #     FULL_MODEL="swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K"
    #     ;;
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

# 7. set port and choice model
port=`expr $RANDOM % 10000 + 20000`
file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

# 8. run model
srun -p $1 -n$2\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} --options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
