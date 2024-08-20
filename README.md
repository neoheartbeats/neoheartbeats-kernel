# Neoheartbeats Kernel

## Current progress

sthenno-gm-04 is a fine-tuned version of DeepMind's gemma2-9b-it.

This model is optimized by KTO(Kahneman-Tversky Optimization) using custom data.

This model is designed to output more naturally that to align human's preferences,
but NOT including to instruct the model to generate human-like outputs such as emotions.

One part of this design is to discover how LLMs implement mental models for
continual-learning and long-term memory's constructions.

Model's safetensors and training data have NOT been disclosed yet but planned to be by
publishing to platforms such as HuggingFace once reliable data is collected under
replicated evaluations.

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
