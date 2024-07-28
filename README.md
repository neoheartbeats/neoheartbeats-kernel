# Tukuyomi (月詠)

<p align="center">
  <img width="20%" src="./images/rpcd0015_main_obj_1.png" />
</p>

Icon source: https://riparia-rec.com/release/rpcd0015/

*Fairy tales of building an AGI*

---
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
最优化 CUDA 目前需要配置系统的 grub, 但远程环境不能直接进 BIOS, 所以目前在配置 Supermicro 的 SUM/BMC, 当作服务器硬件监控使用.

---

### 02 Enable Docker containers

This is specifically for deployment of Qdrant.

### 03 Python scripts

- Transformers/Unsloth for model training
- Optimizing LLM using RAG and continuing data-generating using algorithms like DPO and alternatives like KPO

---

## Appendix：Hardware limiting

- NVIDIA A40 48GB (training and inferences)
- Apple M3 MAX 48GB (inferences)