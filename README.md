# PyTorch 2.11.0 for ARM64 + NVIDIA T4G (SM_75) + CUDA 12.8

NVIDIA T4G (AWS Graviton + Turing) 向けにソースビルドした PyTorch wheel。
公式配布バイナリでは SM_75 が有効化されていないため、T4G 上での動作を保証する専用ビルドです。

---

## 成果物

| 項目 | 値 |
|---|---|
| ファイル | `torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl` |
| サイズ | 167 MB |
| SHA256 | `659816dcc0ce9b0fa6ff8b2f8ff20a76944f37c3106a6d98eb36544ead6b82f9` |
| PyTorch | 2.11.0 (tag `v2.11.0`, commit `70d99e9`) |

---

## 動作条件

### ハードウェア

| 項目 | 要件 |
|---|---|
| CPU | ARM64 / aarch64（AWS Graviton2 相当） |
| GPU | **NVIDIA T4G** (Compute Capability **7.5 / SM_75** のみ) |

> **注意:** `TORCH_CUDA_ARCH_LIST=7.5` でビルドしているため、**他の GPU 世代（A100, H100, Jetson, etc.）では動作しません**。

### ソフトウェア

| 項目 | バージョン | 備考 |
|---|---|---|
| OS | Ubuntu 24.04 LTS (Noble) | glibc 2.39 以上 |
| Kernel | Linux aarch64 | 6.x 系で確認済み |
| NVIDIA Driver | **595.58.03 以上** | CUDA 12.8 要件 |
| CUDA Runtime | **12.8** | ランタイムは wheel に同梱されないため別途インストール必要 |
| cuDNN | 9.x for CUDA 12 | 9.20 で確認済み |
| Python | **3.12** | cp312 ABI 固定 |

### 依存 CUDA パッケージ（Ubuntu apt）

NVIDIA 公式 SBSA リポジトリ (`cuda-ubuntu2404-sbsa`) から：

```bash
sudo apt install -y \
    cuda-toolkit-12-8 \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12
```

ドライバは `nvidia-driver` / `cuda-drivers` (>= 595.58.03)。

---

## インストール

### uv / pip で URL 直接指定

```toml
# pyproject.toml
[project]
dependencies = [
    "torch @ https://github.com/USER/REPO/releases/download/v2.11.0-arm64-sm75-cu128/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl",
]
```

```bash
# pip
pip install "torch @ https://github.com/USER/REPO/releases/download/v2.11.0-arm64-sm75-cu128/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl"

# uv
uv add "torch @ https://github.com/USER/REPO/releases/download/v2.11.0-arm64-sm75-cu128/torch-2.11.0a0+git70d99e9-cp312-cp312-linux_aarch64.whl"
```

---

## 動作確認

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

### 期待される出力

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

## ビルド構成

| 項目 | 値 |
|---|---|
| `TORCH_CUDA_ARCH_LIST` | `7.5` |
| `USE_CUDA` | 1 |
| `USE_CUDNN` | 1 |
| `USE_DISTRIBUTED` | 1 |
| `USE_NCCL` | 0 |
| `USE_MKLDNN` | 0 |
| コンパイラ | gcc 13.3 (Ubuntu 24.04 標準) |
| CMake | 4.3.1 |

---

## 制限事項

- **NCCL 無効**: 複数 GPU / 分散学習時の NCCL backend は使えません。単機シングル GPU 用途を想定。
- **MKLDNN 無効**: aarch64 の CPU 推論は OpenBLAS / XNNPACK / NNPACK 経由。
- **SM_75 以外未対応**: A10G / A100 / H100 / Jetson 等では起動に失敗、または kernel 未実装エラーになります。
- **Python 3.12 固定**: cp312 ABI のみ。3.11 / 3.13 では使えません。
- **aarch64 固定**: x86_64 では使えません。

---

## 周辺エコシステムについて

CUDA 12.8 系のため、以下のライブラリは別途バージョン整合性を確認してください：

- `torchvision`, `torchaudio` — 公式バイナリは CUDA 12.8 / aarch64 向けが限定的。同様にソースビルドが必要な場合あり
- `xformers`, `flash-attn`, `bitsandbytes` — CUDA 12.8 対応版またはソースビルドが必要
- `triton` — ARM64 サポート状況を要確認

---

## ライセンス

PyTorch 本体のライセンス（BSD-3-Clause）に従います。本 wheel は PyTorch 公式ソース (`https://github.com/pytorch/pytorch` tag `v2.11.0`) をそのままビルドしたものです。
