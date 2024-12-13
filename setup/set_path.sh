# ----------------<INPUT required>----------------
export HOME=""
export DATA_DIR=$HOME/data
export SAVE_DIR=$HOME/ckpt
export CACHE_DIR=$HOME/cache
export CODE_DIR=$HOME/LLaVA-Reasoner-DPO
export NOTEBOOK_DIR=$CODE_DIR/notebook

# huggingface token, for data, ckpt downloading
export HF_TOKEN=""
export WANDB_API_KEY=""

# -----<openai key>------
# OPENAI config
export OPENAI_KEY=""
export OPENAI_API_KEY=$OPENAI_KEY # alias
export ORGANIZATION_KEY="" # optional if you have
# -----</openai key>------

# -----<azure key>------
# [Alternatively] for azure openai backend, 
export API_VERSION=""
export AZURE_ENDPOINT=""
export AZURE_OPENAI_KEY=""
# -----</azure key>------

# Export usage, choice [openai, azure]
export OPENAI_BACKEND='openai'
export GPTO_MODEL_NAME="gpt-4o-2024-05-13" # version used at development
export GPTV_MODEL_NAME="gpt-4-turbo-2024-04-09" # version used at development
export GPT_MODEL_NAME="gpt-3.5-turbo-1106" # version used at development
# ----------------</INPUT required>----------------

mkdir -p $DATA_DIR
mkdir -p $SAVE_DIR
mkdir -p $CACHE_DIR

export IMAGE_DATA_DIR=$DATA_DIR/vlm_reason/image_data
export IMAGE_INSTRUCTION_DIR=$DATA_DIR/vlm_reason/image_instruction
export REWARD_DIR=$DATA_DIR/vlm_reason/reward

export RESULT_DIR=${DATA_DIR}/vlm_reason/result
export LMUData=$DATA_DIR/vlm_reason/vlmeval
mkdir -p $LMUData

mkdir -p $IMAGE_DATA_DIR
mkdir -p $IMAGE_INSTRUCTION_DIR

# used for gathering result
export RESULTING_PATH=$RESULT_DIR/result.jsonl

echo use $OPENAI_BACKEND
echo chatgpt model $GPT_MODEL_NAME
echo gptv model $GPTV_MODEL_NAME