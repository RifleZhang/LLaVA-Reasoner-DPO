# Unofficial Repo for LLaVA-Reasoner-DPO
This is an unofficial repo for the paper:
[**Improve Vision Language Model Chain-of-thought Reasoning**](https://arxiv.org/pdf/2410.16198)

## Release
- [10.22] we will provide third party implementation for [arxiv paper](https://arxiv.org/pdf/2410.16198)

## setup
```
# setup environment
source setup/setup_env.sh

# data
source setup/setup_train_data.sh 
```

## sft
```
bash scripts_sft/sft_direct+cot_preview.sh \
$SAVE_DIR/sft/llava_reasoner_sft_preview
```
