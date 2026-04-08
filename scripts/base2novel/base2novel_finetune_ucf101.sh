export NCCL_TIMEOUT=900
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL

cd ../../

model_path=""

VAL_FILE=./configs/base2novel/pretrained_on_k400/ucf101/bmpt_s1.yaml
VAL_FILES=($VAL_FILE)
output_path=./output_ucf/ucf_base2novel_v1
llm_json_path=./prompts/UCF101/ucf101.json

for split_path in "${VAL_FILES[@]}"; do
    PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    -cfg $split_path \
    --output $output_path \
    --llm_json_path $llm_json_path \
    --finetune_fewshot $model_path
done

best_model_path="${output_path}/best.pth"

#Evaluate on base set
VAL_FILE_NOVEL=./configs/base2novel/pretrained_on_k400/ucf101/bmpt_s1.yaml
VAL_FILES=($VAL_FILE_NOVEL)
for split_path in "${VAL_FILES[@]}"; do
    PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 main.py \
     -cfg $split_path \
     --llm_json_path $llm_json_path \
     --finetune_fewshot $best_model_path \
     --only_test
done

#Evaluate on novel set
VAL_FILE_NOVEL=./configs/base2novel/pretrained_on_k400/ucf101/bmpt_novel_eval.yaml
VAL_FILES=($VAL_FILE_NOVEL)
for split_path in "${VAL_FILES[@]}"; do
    PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 main.py \
     -cfg $split_path \
     --llm_json_path $llm_json_path \
     --finetune_fewshot $best_model_path \
     --only_test
done
