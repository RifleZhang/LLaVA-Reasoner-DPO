# Vision Language Model Chain-of-thought Reasoning and Reward
This is an unofficial repo for the paper:
[**Improve Vision Language Model Chain-of-thought Reasoning**](https://arxiv.org/pdf/2410.16198)

## Release
- [12/24 - 01/25] sft, dpo pipeline, distill gpt, inference + eval.
- [10.22] we will provide third party implementation for [arxiv paper](https://arxiv.org/pdf/2410.16198)

## Dataset
[ShareGPT4o-reasoning](https://huggingface.co/datasets/Share4oReasoning/sft_data) 193k cot prediction + filtered direct prediction 
[ShareGPT4o-reasoning-dpo](https://huggingface.co/datasets/Share4oReasoning/dpo_data) 66k DPO data on 3 domains: aokvqa, math and chartqa

## Model ckpt
[Open-LLaVA-NeXT](https://huggingface.co/Share4oReasoning/Open-LLaVA-NeXT-LLaMA3-8B): same as https://github.com/xiaoachen98/Open-LLaVA-NeXT, used as our base model 

[LLaVA-Reasoner-SFT-preview](https://huggingface.co/Share4oReasoning/LLaVA-Reasoner-SFT-preview): SFT with direct + CoT

[LLaVA-Reasoner-SFT](https://huggingface.co/Share4oReasoning/LLaVA-Reasoner-SFT): SFT with direct + CoT (additional math than above)

[LLaVA-Reasoner-DPO-preview](https://huggingface.co/Share4oReasoning/LLaVA-Reasoner-DPO-preview): DPO from SFT-preview


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
## citation
```
@article{zhang2024improve,
  title={Improve vision language model chain-of-thought reasoning},
  author={Zhang, Ruohong and Zhang, Bowen and Li, Yanghao and Zhang, Haotian and Sun, Zhiqing and Gan, Zhe and Yang, Yinfei and Pang, Ruoming and Yang, Yiming},
  journal={arXiv preprint arXiv:2410.16198},
  year={2024}
}
```

## Acknowledge
Thanks to 

(open-llava-next)[https://github.com/xiaoachen98/Open-LLaVA-NeXT]: for base model and sft training

(LLaVA-Hound)[https://github.com/RifleZhang/LLaVA-Hound-DPO/tree/main]: for dpo related 

