# Neoheartbeats Kernel

## (TODO)

---
base_model: /home/ubuntu/models/mistral-small
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: miscii-0918-08
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# miscii-0918-08

This model is a fine-tuned version of [/home/ubuntu/models/mistral-small](https://huggingface.co//home/ubuntu/models/mistral-small) on the kto-12 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2786
- Rewards/chosen: 5.3903
- Logps/chosen: -59.0879
- Rewards/rejected: -6.2351
- Logps/rejected: -169.1946
- Rewards/margins: 11.6255
- Kl: 1.2679

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 8e-05
- train_batch_size: 4
- eval_batch_size: 24
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Rewards/chosen | Logps/chosen | Rewards/rejected | Logps/rejected | Rewards/margins | Kl     |
|:-------------:|:------:|:----:|:---------------:|:--------------:|:------------:|:----------------:|:--------------:|:---------------:|:------:|
| 0.1947        | 1.3115 | 50   | 0.3785          | 3.3771         | -75.8649     | -2.4241          | -137.4358      | 5.8012          | 0.0    |
| 0.1604        | 2.6230 | 100  | 0.3099          | 3.9486         | -71.1022     | -6.3713          | -170.3293      | 10.3199         | 0.1090 |
| 0.0798        | 3.9344 | 150  | 0.2796          | 5.2203         | -60.5045     | -6.5271          | -171.6276      | 11.7474         | 1.1228 |


### Framework versions

- PEFT 0.12.0
- Transformers 4.44.2
- Pytorch 2.4.0+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1

## Current progress

sthenno-gm-05-05 is a fine-tuned version of DeepMind's gemma2-9b-it.

This model is optimized by KTO(Kahneman-Tversky Optimization) using custom data.

This model is designed to output more naturally that to align human's preferences,
but NOT including to instruct the model to generate human-like outputs such as emotions.

One part of this design is to discover how LLMs implement mental models for
continual-learning and long-term memory's constructions.

Model's safetensors and training data have NOT been disclosed yet but planned to be by
publishing to platforms such as HuggingFace once reliable data is collected under
replicated evaluations.

### Training Arguments

- Training device: NVIDIA A40
- Memory usage: up to 46GB
- Framework used: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Base model: [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)

```yaml
bf16: true
cutoff_len: 1024
dataset: kto-04
dataset_dir: data
ddp_timeout: 180000000
do_train: true
finetuning_type: lora
gradient_accumulation_steps: 8
include_num_input_tokens_seen: true
learning_rate: 8.0e-05
lora_alpha: 32
lora_dropout: 0
lora_rank: 16
lora_target: all
lr_scheduler_type: cosine
max_grad_norm: 1.0
max_samples: 3000
model_name_or_path: /home/neoheartbeats/endpoint/models/gm2-9b-it
num_train_epochs: 120.0
optim: adamw_torch
output_dir: saves/Gemma-2-9B-Chat/lora/gm-005-05
packing: false
per_device_train_batch_size: 4
plot_loss: true
pref_beta: 0.06
pref_ftx: 0
pref_loss: kto_pair
stage: kto
template: gemma
```

![training_loss](./images/training_loss.png)

## Roadmap
### 01 Optimize CUDA kernels

- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/release-notes.html

- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda

- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html

- https://docs.docker.com/config/containers/resource_constraints/

- https://www.supermicro.com/en/support/resources/downloadcenter/smsdownload?category=SUM

- https://docs.portainer.io/v/2.20/start/upgrade/docker

---

现在第一步任务是部署一个服务端的向量数据库 (当前选择 Qdrant),
使用 CUDA 开发版 (并非企业部署), 在 Docker 和 Conda 环境下启用.
最优化 CUDA 目前需要配置系统的 grub, 但远程环境不能直接进 BIOS,
所以目前在配置 Supermicro 的 SUM/BMC, 当作服务器硬件监控使用.

---

### 02 Enable Docker containers

This is specifically for deployment of Qdrant.

### 03 Python scripts

- Transformers/Unsloth for model training
- Optimizing LLM using RAG and continuing data-generating using algorithms like DPO and
alternatives like KTO

---

## Appendix：Hardware limiting

- NVIDIA A40 48GB (training and inferences)
- Apple M3 MAX 48GB (inferences)
