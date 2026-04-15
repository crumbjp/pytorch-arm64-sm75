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
| cuDNN | 9.x for CUDA 12 | Verified with 9.20 / 9.21 (any 9.x for CUDA 12 should work) |
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

## Runtime Environment Setup (from a fresh Ubuntu 24.04 g5g instance)

The following sequence prepares a fresh AWS g5g instance (Ubuntu 24.04 LTS)
to run this wheel. Run as `root` (or prepend `sudo`).

```bash
# 1. Update apt and install matching kernel headers
apt update
apt install -y linux-headers-$(uname -r)
# Note: an AWS-specific meta-package such as `linux-aws-headers-*` may
# pin an outdated header version that conflicts with the running kernel.
# Only if the install above fails with such a conflict, remove the
# offending package first, e.g.:
#   apt remove -y linux-aws-headers-<old-version>
# then retry the install.

# 2. Add the NVIDIA CUDA SBSA (ARM64 server) apt repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update

# 3. Install CUDA 12.8 toolkit + driver + cuDNN 9 + OpenBLAS
apt install -y cuda-toolkit-12-8
apt install -y cuda-drivers
apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libopenblas0

# 4. Reboot to load the NVIDIA kernel modules
reboot
```

After reboot, verify the GPU is visible:

```bash
nvidia-smi
# Expect to see "NVIDIA T4G" with a working driver / CUDA 12.8 runtime.
```

### Python 3.12 environment (via uv)

```bash
# Install uv (per-user)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a project venv pinned to Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install this PyTorch wheel
uv pip install "torch @ https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl"
```

### Notes

- The kernel header step is required so the NVIDIA driver's DKMS module
  can build against the running kernel.
- `cuda-toolkit-12-8` brings in the full toolkit (nvcc, headers, libraries).
  If you only need the runtime (no compilation on the box), `cuda-runtime-12-8`
  is sufficient and smaller.
- `libopenblas0` is needed because this wheel uses OpenBLAS for CPU BLAS
  operations (MKLDNN is disabled on aarch64).

---

## Installation

### Quick install with pip / uv

```bash
# pip
pip install "torch @ https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl"

# uv
uv add "torch @ https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl"
```

### `pyproject.toml` examples

#### Minimal (PEP 508 direct URL)

```toml
[project]
name = "your-project"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch @ https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl",
]
```

#### uv with platform-conditional dependency

If your `pyproject.toml` is shared between aarch64 (g5g) and other
architectures, gate the wheel on the platform so it only resolves on
Linux aarch64:

```toml
[project]
name = "your-project"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    # On AWS g5g (Linux aarch64) — use this prebuilt wheel.
    "torch @ https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl ; sys_platform == 'linux' and platform_machine == 'aarch64'",
    # On other platforms — use the official PyPI build.
    "torch ; sys_platform != 'linux' or platform_machine != 'aarch64'",
]
```

#### uv with `[tool.uv.sources]` (recommended for uv users)

```toml
[project]
name = "your-project"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch",
]

[tool.uv.sources]
torch = { url = "https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl" }
```

To restrict the override to aarch64 only (keeping PyPI for other platforms):

```toml
[tool.uv.sources]
torch = [
    { url = "https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl",
      marker = "sys_platform == 'linux' and platform_machine == 'aarch64'" },
]
```

#### Poetry

```toml
[tool.poetry.dependencies]
python = ">=3.12,<3.13"
torch = { url = "https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl" }
```

#### `requirements.txt` (pip)

```
torch @ https://github.com/crumbjp/pytorch-arm64-sm75/releases/download/v2.11.0-cu128-cp312/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl
```

---

## Verification

> **Tip:** PyTorch itself does not pull in NumPy as a hard dependency,
> but most ML workflows do. If you see
> `UserWarning: Failed to initialize NumPy: No module named 'numpy'`
> on import, install it explicitly:
>
> ```bash
> uv pip install numpy
> ```

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
cudnn version: 92100        # any 9.x for CUDA 12; e.g. 92000 (9.20), 92100 (9.21)
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
