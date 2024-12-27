model_name=$SAVE_DIR/hf/open-llava-next-llama3-8b
output_dir=${1:-"$SAVE_DIR/sft/LLaVA-Reasoner-SFT-preview"}

sft_dir=$IMAGE_INSTRUCTION_DIR/sft
pretrain_dir=$IMAGE_INSTRUCTION_DIR/pretrain

# data composition
# pretrain 2k mix + direct + cot
data_paths="\
$sft_dir/pt_data_mix_2k.jsonl \
$sft_dir/docvqa/direct.none.jsonl \
$sft_dir/infovqa/direct.none.jsonl \
$sft_dir/chartqa/direct.none.jsonl \
$sft_dir/ai2d/direct.none.jsonl \
$sft_dir/aokvqa/direct.none.jsonl \
$sft_dir/sqa/direct.image.none.jsonl \
$sft_dir/textvqa/direct.none.jsonl \
$sft_dir/docvqa/cot.none.jsonl \
$sft_dir/infovqa/cot.none.jsonl \
$sft_dir/chartqa/cot.none.jsonl \
$sft_dir/ai2d/cot.none.jsonl \
$sft_dir/aokvqa/cot.none.jsonl \
$sft_dir/sqa/cot.image.none.jsonl \
$sft_dir/textvqa/cot.none.jsonl \
$sft_dir/math/gllava.align.jsonl \
$sft_dir/math/geoqa_plus.qa.jsonl \
$sft_dir/mathvision/cot.none.jsonl \
$sft_dir/math/geo.mcq.x2.jsonl \
$sft_dir/math/geo.sa.x2.jsonl \
$sft_dir/math/geo-align.rationale.4k.jsonl \
$sft_dir/math/geo-qa.rationale.4k.jsonl \
$sft_dir/math/geo-align.qa.4k.jsonl \
$sft_dir/math/geo-qa.qa.4k.jsonl \
"

export TOKENIZERS_PARALLELISM=false
gpu_ids=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"
export PYTHONPATH=.
rand=$RANDOM
port=$((19000 + $rand % 1000))

cache_dir=$CACHE_DIR
export cache_dir=$cache_dir

image_folder=$IMAGE_DATA_DIR

save_name=$(basename $output_dir)
export WANDB_PROJECT=llava-llama3-reasoning
export WANDB_NAME=${save_name}
export report_to=wandb
wandb_args="--report_to $report_to"

echo input model: $model_name
echo output model: $output_dir
mkdir -p $output_dir

# tokenizer_name=$model_name_or_path
version=llava_llama_3

# pretrain
# total batch 4 node * 8 gpu * 4 batch = 128
# BASE_LR=2e-5
# VIT_LR=2e-6

# sft
batch_size=4
grad_cum=1
BASE_LR=5e-6
VIT_LR=5e-7

# continue train doesn't require
# --pretrain_mm_mlp_adapter checkpoints/llava-v1.6-8b_llama3-8b_pretrain_lcs-558k_ft-mlp-lr-1e-3/mm_projector.bin \
# --save_only_model True

torchrun --nproc_per_node=$n_gpu --master_port=$port llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --unfreeze_mm_vision_tower True --mm_vision_tower_lr ${VIT_LR} \
    --model_name_or_path $model_name \
    --version $version \
    --data_paths $data_paths \
    --image_folder $image_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio anyres \
    --group_by_modality_length True --dataloader_drop_last True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type spatial_unpad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${grad_cum} \
    --evaluation_strategy "no" \
    --save_strategy "steps" --save_steps 0.333 --save_total_limit 1 --save_only_model True \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 6144 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    ${wandb_args} 2>&1 | tee $output_dir/train.log
