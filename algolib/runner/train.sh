#!/bin/bash
set -x

if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

# 0. build soft link for mm configs
if [ -x "$SMART_ROOT/submodules" ];then
    submodules_root=$SMART_ROOT
else    
    submodules_root=$PWD
fi

if [ -d "$submodules_root/submodules/mmseg/algolib/configs" ]
then
    rm -rf $submodules_root/submodules/mmseg/algolib/configs
    ln -s $submodules_root/submodules/mmseg/configs $submodules_root/submodules/mmseg/algolib/
else
    ln -s $submodules_root/submodules/mmseg/configs $submodules_root/submodules/mmseg/algolib/
fi
 
# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmseg/$3
export PYTORCH_VERSION=1.4
 
# 2. set time
now=$(date +"%Y%m%d_%H%M%S")
 
# 3. set env
path=$PWD
if [[ "$path" =~ "submodules" ]]
then 
    pyroot=$submodules_root/mmseg
else
    pyroot=$submodules_root/submodules/mmseg
fi
echo $pyroot
export PYTHONPATH=$pyroot:$PYTHONPATH
export FRAME_NAME=mmseg    #customize for each frame
export MODEL_NAME=$3

# 4. set init_path
export PYTHONPATH=$SMART_ROOT/common/sites/:$PYTHONPATH 
 
# 5. build necessary parameter
partition=$1 
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
EXTRA_ARGS=${array[@]:3}
EXTRA_ARGS=${EXTRA_ARGS//--resume/--resume-from}
SRUN_ARGS=${SRUN_ARGS:-""}
 
# 6. model choice
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

port=`expr $RANDOM % 10000 + 20000`

file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

srun -p $1 -n$2\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} --options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
