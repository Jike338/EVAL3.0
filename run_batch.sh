python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --dataset_split testmini \
    --model_name_path OpenGVLab/InternVL3-8B-hf \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_engine hf \
    --tag none

python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --dataset_split testmini \
    --model_name_path OpenGVLab/InternVL3-8B-hf \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type "Please provide the answer first, and then, explain your reasoning." \
    --gen_engine hf \
    --tag ans_then_reason

python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --dataset_split testmini \
    --model_name_path OpenGVLab/InternVL3-8B-hf \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type  "Please try to answer the question with short words or phrases if possible." \
    --gen_engine hf \
    --tag qwen

python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --dataset_split testmini \
    --model_name_path OpenGVLab/InternVL3-8B-hf \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type "Please think step by step and output answer at the end." \
    --gen_engine hf \
    --tag step_cot