# Unofficial Repo for LLaVA-Reasoner-DPO
This is an unofficial repo for the paper:
[**Improve Vision Language Model Chain-of-thought Reasoning**](https://arxiv.org/pdf/2410.16198)

## Release
- [12/24 - 01/25] sft, dpo pipeline, distill gpt, inference + eval.
- [10.22] we will provide third party implementation for [arxiv paper](https://arxiv.org/pdf/2410.16198)

## Dataset
[ShareGPT4o-reasoning](https://huggingface.co/datasets/Share4oReasoning/sft_data) 193k cot prediction + filtered direct prediction 

## Model ckpt


## setup 
```
# setup environment, need to fill in the required fields
source setup/setup_env.sh

# data
source setup/setup_train_data.sh 
```

## sft
```
cd llava_reasoner
bash scripts_sft/sft_direct+cot_preview.sh \
$SAVE_DIR/sft/LLaVA-Reasoner-SFT-preview
```

## dpo
```
cd llava_reasoner
bash scripts_dpo/dpo_llava_reasoner_preview.sh \
$SAVE_DIR/dpo/LLaVA-Reasoner-DPO-preview
```


