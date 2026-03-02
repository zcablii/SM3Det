set -x

CHECKPOINT=${1}
DATASET=${2}
CHECKPOINT="$(pwd)/${CHECKPOINT}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

# Save original arguments
ARGS=("$@")

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
echo "GPUS: ${GPUS}"


if [ ${DATASET} == "mcq_test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets mcq_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_sar_Sentinel" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_sar_Sentinel "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_UCM" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_UCM "${ARGS[@]:2}"
fi
if [ ${DATASET} == "cls_fmow" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_fmow "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_sar_ISPRS" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_sar_ISPRS "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_sar_TenGeoP" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_sar_TenGeoP "${ARGS[@]:2}"
fi


if [ ${DATASET} == "dior-rsvg" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_det/evaluate.py --checkpoint ${CHECKPOINT} --datasets DIOR_RSVG "${ARGS[@]:2}"
fi

if [ ${DATASET} == "geochat" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_det/evaluate.py --checkpoint ${CHECKPOINT} --datasets geochat "${ARGS[@]:2}"
fi

if [ ${DATASET} == "rsvg" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_det/evaluate.py --checkpoint ${CHECKPOINT} --datasets rsvg "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vrsbench" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_det/evaluate.py --checkpoint ${CHECKPOINT} --datasets vrsbench "${ARGS[@]:2}"
fi


# if [ ${DATASET} == "dior-rsvg" ]; then
#     torchrun \
#       --nnodes=1 \
#       --node_rank=0 \
#       --master_addr=127.0.0.1 \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       eval/domain_specific/rs_vqa/evaluate.py --checkpoint ${CHECKPOINT} --datasets RSVQA_H_TEST2 "${ARGS[@]:2}"
# fi

if [ ${DATASET} == "rsvqa-lr" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_vqa/evaluate.py --checkpoint ${CHECKPOINT} --datasets RSVQA_H_TEST2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "rsvqa-hr-test1" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_vqa/evaluate.py --checkpoint ${CHECKPOINT} --datasets RSVQA_H_TEST1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "rsvqa-hr-test2" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_vqa/evaluate.py --checkpoint ${CHECKPOINT} --datasets RSVQA_L "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_AID" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_AID "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_AID_100" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_AID_100 "${ARGS[@]:2}"
fi


if [ ${DATASET} == "cls_AID_1" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_AID_1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_METER_ML" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_METER_ML "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_NWPU_RESISC45" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_NWPU_RESISC45 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_SIRI_WHU" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_SIRI_WHU "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_WHU_RS19" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_WHU_RS19 "${ARGS[@]:2}"
fi
if [ ${DATASET} == "cls_WHU_RS19_hard" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_WHU_RS19_hard "${ARGS[@]:2}"
fi

if [ ${DATASET} == "cls_AID_CoT" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_AID "${ARGS[@]:2}" --cot
fi


if [ ${DATASET} == "cls_METER_ML_CoT" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_METER_ML "${ARGS[@]:2}" --cot
fi

if [ ${DATASET} == "cls_NWPU_RESISC45_CoT" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_NWPU_RESISC45 "${ARGS[@]:2}" --cot
fi

if [ ${DATASET} == "cls_SIRI_WHU_CoT" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_SIRI_WHU "${ARGS[@]:2}" --cot
fi

if [ ${DATASET} == "cls_WHU_RS19_CoT" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_WHU_RS19 "${ARGS[@]:2}" --cot
fi

if [ ${DATASET} == "cls_millionAID" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_millionAID "${ARGS[@]:2}" 
fi

if [ ${DATASET} == "cls_millionAID_CoT" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vhm/evaluate_vhm.py --checkpoint ${CHECKPOINT} --datasets cls_millionAID "${ARGS[@]:2}" --cot
fi



if [ ${DATASET} == "DOTA" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb.py --checkpoint ${CHECKPOINT} --datasets DOTA "${ARGS[@]:2}"
fi

if [ ${DATASET} == "FAIR1M2" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb.py --checkpoint ${CHECKPOINT} --datasets FAIR1M2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "RSAR" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb.py --checkpoint ${CHECKPOINT} --datasets RSAR "${ARGS[@]:2}"
fi

if [ ${DATASET} == "SRSDD" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb.py --checkpoint ${CHECKPOINT} --datasets SRSDD "${ARGS[@]:2}"
fi

if [ ${DATASET} == "DOTA_100" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb.py --checkpoint ${CHECKPOINT} --datasets DOTA_100 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "MillionAID" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb.py --checkpoint ${CHECKPOINT} --datasets MillionAID "${ARGS[@]:2}"
fi

if [ ${DATASET} == "dior-rsvg_100" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/domain_specific/rs_det/evaluate.py --checkpoint ${CHECKPOINT} --datasets DIOR_RSVG_100 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "DOTA_multiround" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb_multiround.py --checkpoint ${CHECKPOINT} --datasets DOTA "${ARGS[@]:2}"
fi

if [ ${DATASET} == "FAIR1M2_multiround" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb_multiround.py --checkpoint ${CHECKPOINT} --datasets FAIR1M2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "RSAR_multiround" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb_multiround.py --checkpoint ${CHECKPOINT} --datasets RSAR "${ARGS[@]:2}"
fi

if [ ${DATASET} == "SRSDD_multiround" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb_multiround.py --checkpoint ${CHECKPOINT} --datasets SRSDD "${ARGS[@]:2}"
fi

if [ ${DATASET} == "Vaihingen" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seg/eval_seg.py --checkpoint ${CHECKPOINT} --datasets Vaihingen "${ARGS[@]:2}"
fi


if [ ${DATASET} == "SARdet_vis" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/hbb/vis.py --checkpoint ${CHECKPOINT} --dataset SARdet "${ARGS[@]:2}"
fi

if [ ${DATASET} == "GAIA" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --dataset GAIA "${ARGS[@]:2}"
fi

if [ ${DATASET} == "CapERA" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/caption/evaluate_video_caption.py --checkpoint ${CHECKPOINT} --dataset CapERA "${ARGS[@]:2}"
fi

if [ ${DATASET} == "DOTA_objectperround" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/obb/evaluate_obb_objectperround.py --checkpoint ${CHECKPOINT} --dataset DOTA "${ARGS[@]:2}"
fi