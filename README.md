# PyTorch 2.11.0 for AWS g5g (ARM64 + NVIDIA T4G / SM_75 + CUDA 12.8)

A PyTorch wheel built from source for **AWS EC2 `g5g` instances**
(Graviton2 + NVIDIA T4G GPU).

The official PyTorch binaries do not enable `SM_75` for the aarch64 builds,
so this wheel is provided to ensure proper operation of the T4G GPU on g5g.

---

## Target Environment

| Item | Value |
|---|---|
| Instance type | **AWS EC2 `g5g.*`** (g5g.xlarge / 2xlarge / 4xlarge / 8xlarge / 16xlarge / metal) |
| CPU | AWS Graviton2 (aarch64) |
| GPU | NVIDIA T4G — Compute Capability **7.5 / SM_75** |
| OS | Ubuntu 24.04 LTS (Noble) |

---

## Artifact

| Item | Value |
|---|---|
| File | `torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl` |
| Size | 167 MB |
| SHA256 | `659816dcc0ce9b0fa6ff8b2f8ff20a76944f37c3106a6d98eb36544ead6b82f9` |
| PyTorch | 2.11.0 (tag `v2.11.0`, commit `70d99e9`) |

---

## Requirements

### Hardware

| Item | Requirement |
|---|---|
| CPU | ARM64 / aarch64 (AWS Graviton2 or compatible) |
| GPU | **NVIDIA T4G** — Compute Capability **7.5 / SM_75** only |

> **Note:** This wheel is built with `TORCH_CUDA_ARCH_LIST=7.5`.
> It will **not work** on other GPU generations (A10G, A100, H100, Jetson, etc.).

### Software

| Item | Version | Notes |
|---|---|---|
| OS | Ubuntu 24.04 LTS (Noble) | glibc >= 2.39 |
| Kernel | Linux aarch64 | Tested on 6.x |
| NVIDIA Driver | **>= 595.58.03** | Required by CUDA 12.8 |
| CUDA Runtime | **12.8** | Not bundled in the wheel; install separately |
| cuDNN | 9.x for CUDA 12 | Tested with 9.20 |
| Python | **3.12** | cp312 ABI only |

### Required Ubuntu apt packages

From the official NVIDIA SBSA repository (`cuda-ubuntu2404-sbsa`):

```bash
sudo apt install -y \
    cuda-toolkit-12-8 \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12
```

Driver: `nvidia-driver` / `cuda-drivers` (>= 595.58.03).

---

## Installation

### Direct URL with uv / pip

```toml
# pyproject.toml
[project]
dependencies = [
    "torch @ https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl",
]
```

```bash
# pip
pip install "torch @ https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl"

# uv
uv add "torch @ https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl"
```

---

## Verification

```python
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())
print("device name:", torch.cuda.get_device_name(0))
print("device capability:", torch.cuda.get_device_capability(0))

x = torch.randn(256, 256, device="cuda")
y = x @ x
print("matmul ok:", y.shape, y.device)
```

### Expected output

```
torch: 2.11.0a0+git70d99e9
cuda available: True
cuda version: 12.8
cudnn version: 92000
device name: NVIDIA T4G
device capability: (7, 5)
matmul ok: torch.Size([256, 256]) cuda:0
```

---

## Build Configuration

| Item | Value |
|---|---|
| `TORCH_CUDA_ARCH_LIST` | `7.5` |
| `USE_CUDA` | 1 |
| `USE_CUDNN` | 1 |
| `USE_DISTRIBUTED` | 1 |
| `USE_NCCL` | 0 |
| `USE_MKLDNN` | 0 |
| Compiler | gcc 13.3 (Ubuntu 24.04 default) |
| CMake | 4.3.1 |

---

## Limitations

- **NCCL disabled**: The NCCL backend for multi-GPU / distributed training is not available. Intended for single-host single-GPU workloads.
- **MKLDNN disabled**: CPU inference paths use OpenBLAS / XNNPACK / NNPACK on aarch64.
- **SM_75 only**: Will fail to launch (or hit "no kernel image" errors) on A10G / A100 / H100 / Jetson and other architectures.
- **Python 3.12 only**: cp312 ABI exclusively. Not usable with 3.11 or 3.13.
- **aarch64 only**: Not usable on x86_64.

---

## Ecosystem Notes

Because this targets CUDA 12.8, please verify version compatibility for related libraries:

- `torchvision`, `torchaudio` — Official aarch64 + CUDA 12.8 binaries are limited; you may also need to build from source.
- `xformers`, `flash-attn`, `bitsandbytes` — Require CUDA 12.8-compatible builds or source builds.
- `triton` — Verify aarch64 support status.

---

## License

Subject to the PyTorch license (BSD-3-Clause). This wheel is an unmodified
build of the official PyTorch source at `https://github.com/pytorch/pytorch`
tag `v2.11.0`.
