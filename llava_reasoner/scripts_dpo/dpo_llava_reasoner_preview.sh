# model_name_or_path=$SAVE_DIR/upload/LLaVA-Reasoner-SFT-preview
model_name_or_path=$SAVE_DIR/sft/0921_sft_A_math-text+math2
# model_path=
output_dir=${1:-"$SAVE_DIR/dpo/LLaVA-Reasoner-DPO-preview"}
truncate_len=${2:-90} # > 1 (i.e. 90) is truncate length, < 1 (i.e. 0.8) is truncate ratio
# discount_factor_type supports ['cos', 'linear'] decay for dpo token reweighting. 'none' for no discount factor (default).
# cos decay has similar performance to truncate. linear decay is worse.

# DATA
dpo_dir=$IMAGE_INSTRUCTION_DIR/dpo

data_paths="\
$dpo_dir/chartqa.pairwise.train.jsonl
$dpo_dir/aokvqa.pairwise.train.jsonl
"
eval_data_paths="\
$dpo_dir/chartqa.pairwise.dev.jsonl
$dpo_dir/aokvqa.pairwise.dev.jsonl
"

echo data paths: $data_paths
echo eval data paths: $eval_data_paths


export TOKENIZERS_PARALLELISM=false

cache_dir=$CACHE_DIR
export cache_dir=$cache_dir

# export WANDB_MODE=disabled
# export report_to=none
save_name=$(basename $output_dir)
export WANDB_PROJECT=llava-llama3-dpo
export WANDB_NAME=${save_name}
export report_to=wandb
wandb_args="--report_to $report_to"


gpu_ids=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"
export PYTHONPATH=.
rand=$RANDOM
port=$((19000 + $rand % 1000))

echo input model: $model_name_or_path
echo output model: $output_dir
mkdir -p $output_dir


image_folder=$IMAGE_DATA_DIR

version=llava_llama_3
BASE_LR=5e-7
batch_size=1
grad_cum=4


# not updating vision tower, skip:
# --mm_vision_tower_lr ${VIT_LR} \
# only vision skip:
# --group_by_modality_length True \

# train 1 ep + truncate, freeze mm_projector
torchrun --nproc_per_node=$n_gpu --master_port=$port scripts_dpo/run_dpo.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $model_name_or_path \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --truncate_len $truncate_len --discount_factor_type none \
    --unfreeze_mm_vision_tower False --freeze_mm_mlp_adapter True \
    --version $version \
    --data_paths ${data_paths} --eval_data_paths ${eval_data_paths} \
    --image_folder $image_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio anyres \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type spatial_unpad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps $grad_cum \
    --evaluation_strategy "steps" --eval_steps 0.1 \
    --save_strategy "no"  --save_only_model True --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 6144 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none \
    --dataloader_num_workers 8 \
    --cache_dir $cache_dir \
    --report_to $report_to 2>&1 | tee $output_dir/train.log
