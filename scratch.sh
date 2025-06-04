export VLLM_WORKER_MULTIPROC_METHOD=spawn

python3 main.py \
    --dataset_name_path /home/jikezhong/EVAL3.0/custom_datasets/mix_data/mix_task_v1_jike.json \
    --dataset_dir /home/jikezhong/EVAL3.0/custom_datasets/mix_data/mix_data \
    --model_name_path Qwen/Qwen2.5-VL-7B-Instruct \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type cot \
    --gen_engine vllm \
    --tag mix_cot \
    --n_generations 5\
    --temperature 1\
    --debug

python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --dataset_split testmini \
    --model_name_path OpenGVLab/InternVL3-9B \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type dir \
    --gen_engine vllm \
    --task_name mathvista \
    --tag dir \
    --debug


python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --dataset_split testmini \
    --model_name_path OpenGVLab/InternVL3-9B \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type dir \
    --gen_engine hf \
    --task_name mathvista \
    --tag dir \
    --debug

    
llava-hf/llava-v1.6-mistral-7b-hf
llava-hf/llava-v1.6-vicuna-7b-hf


python3 main.py \
    --dataset_name_path /scratch1/jikezhon/R1-V/src/eval/prompts/superclevr_test200_counting_problems.jsonl \
    --dataset_dir /scratch1/jikezhon/R1-V/src/eval/images \
    --model_name_path llava-hf/llava-v1.6-vicuna-7b-hf \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type dir \
    --gen_engine hf \
    --tag dir_t1n1 \
    --debug
    

python3 main.py \
    --dataset_name_path /scratch1/jikezhon/R1-V/src/eval/prompts/superclevr_test200_counting_problems.jsonl \
    --dataset_dir /scratch1/jikezhon/R1-V/src/eval/images \
    --model_name_path /scratch1/jikezhon/Visual-RFT/res/Qwen2-VL-2B-Instruct/checkpoint-100 \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type cot_tag \
    --gen_engine vllm \
    --debug

python3 main.py \
    --dataset_name_path /scratch1/jikezhon/R1-V/src/eval/prompts/superclevr_test200_counting_problems.jsonl \
    --dataset_dir /scratch1/jikezhon/R1-V/src/eval/images \
    --model_name_path Qwen/Qwen2-VL-2B-Instruct \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type cot_tag \
    --gen_engine hf \
    --debug


python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --model_name_path Qwen/Qwen2-VL-2B-Instruct \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type cot_tag \
    --gen_engine hf \
    --debug \
    --dataset_split testmini


python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --dataset_split testmini \
    --model_name_path Qwen/Qwen2.5-VL-7B-Instruct \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type dir \
    --gen_engine vllm \
    --tag hehe
    
python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --dataset_split testmini \
    --model_name_path Qwen/Qwen2-VL-2B-Instruct \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type dir \
    --gen_engine vllm \
    --task_name mathvista \
    --tag dir

python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --dataset_split testmini \
    --model_name_path Qwen/Qwen2.5-VL-3B-Instruct \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix_type "Please think step by step and output answer at the end." \
    --gen_engine vllm \
    --tag step_cot


python3 main.py \
    --dataset_name_path AI4Math/MathVista \
    --dataset_split testmini \
    --model_name_path Qwen/Qwen2-VL-2B-Instruct \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --delete_prev_file \
    --bs 8 \
    --gen_prompt_suffix cot \
    --gen_engine hf \
    --task_name mathvista \
    --tag cot_hf\
    --debug


python3 main.py \
    --duty_calc_score \
    --file_with_extracted_response /scratch1/jikezhon/EVAL/results/mathvista/Qwen2-VL-2B-Instruct_extract_cot.json


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


python3 main.py     --dataset_name_path AI4Math/MathVista     --dataset_split testmini     --model_name_path Qwen/Qwen2-VL-2B-Instruct     --bs 8     --gen_engine hugginface --debug --do_generate_raw_response --do_extract_answer --task_name mathvista \
--api_key  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
