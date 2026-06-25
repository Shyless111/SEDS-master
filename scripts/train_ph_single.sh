DATA_PATH=""
TIME_NOW=$(date +%Y%m%d%H%M%S)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29533}"
RUN_TAG="${RUN_TAG:-default}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-/sda/home/shihaoyu/Projects/MESM/distilbert-base-uncased}"
USE_UATVR_HEAD="${USE_UATVR_HEAD:-0}"
UATVR_USE_DSL="${UATVR_USE_DSL:-0}"
UATVR_DSL_MODE="${UATVR_DSL_MODE:-col}"
UATVR_MIL_WEIGHT="${UATVR_MIL_WEIGHT:-1e-2}"
UATVR_KL_WEIGHT="${UATVR_KL_WEIGHT:-1e-4}"
N_VIDEO_EMBEDDINGS="${N_VIDEO_EMBEDDINGS:-7}"
N_TEXT_EMBEDDINGS="${N_TEXT_EMBEDDINGS:-7}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUTPUT_DIR="result_train/ph_video_text_single_bs${BATCH_SIZE}_${RUN_TAG}_${TIME_NOW}"

CMD=(python -m torch.distributed.launch --nproc_per_node=1 --master_port "${MASTER_PORT}" \
main_task_retrieval.py --do_train --num_thread_reader="${NUM_WORKERS}" \
--epochs=200 --batch_size="${BATCH_SIZE}" --n_display=10 \
--data_path data_ph \
--features_path "/sda/home/shihaoyu/Projects/SEDS-master/Datasets/PHOENIX-2014-T/features/RTM_Keypoints/" \
--features_RGB_path "/sda/home/shihaoyu/Projects/SEDS-master/Datasets/PHOENIX-2014-T/features/I3D_features/" \
--output_dir "${OUTPUT_DIR}" \
--lr 1e-5 --sign_lr 1e-4 \
--max_words 32 --feature_len 64 --max_length_frames 300 \
--slide_windows 16 --windows_stride 1 \
--crop_size 256 --frames_threshold 0.1 --threshold 0.4 \
--batch_size_val "${BATCH_SIZE_VAL}" \
--datatype ph_pose --coef_lr 1. --freeze_layer_num 0 \
--linear_patch 2d --sim_header Filip --filip_only \
--text_encoder_path "${TEXT_ENCODER_PATH}" \
--pretrained_clip_name ViT-B/32)

if [ "${USE_UATVR_HEAD}" = "1" ]; then
  CMD+=(--use_uatvr_head \
  --uatvr_mil_weight "${UATVR_MIL_WEIGHT}" \
  --uatvr_kl_weight "${UATVR_KL_WEIGHT}" \
  --n_video_embeddings "${N_VIDEO_EMBEDDINGS}" \
  --n_text_embeddings "${N_TEXT_EMBEDDINGS}")
fi

if [ "${UATVR_USE_DSL}" = "1" ]; then
  CMD+=(--uatvr_use_dsl --uatvr_dsl_mode "${UATVR_DSL_MODE}")
fi

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}"
