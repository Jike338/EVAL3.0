#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn

models=(
  # "Qwen/Qwen2-VL-2B-Instruct"
  "Qwen/Qwen2.5-VL-32B-Instruct"
  # "Qwen/Qwen2.5-VL-7B-Instruct"
  # "llava-hf/llava-v1.6-vicuna-7b-hf"
  "OpenGVLab/InternVL3-9B"
)

prompt_types=(
  # "dir"
  "cot"
)

for model in "${models[@]}"
do
  for prompt_type in "${prompt_types[@]}"
  do
    echo "Running with model: $model and prompt type: $prompt_type"
    python3 main.py \
      --dataset_name_path AI4Math/MathVista \
      --dataset_split testmini \
      --model_name_path "$model" \
      --duty_generation \
      --duty_extract_answer \
      --duty_calc_score \
      --delete_prev_file \
      --bs 8 \
      --gen_prompt_suffix_type "$prompt_type" \
      --gen_engine vllm \
      --tag "${prompt_type}"
  done
done
