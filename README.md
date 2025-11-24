# Self-Supervised Tooth-Level Radiograph Foundations for Faster Dental AI Research and Application
**“Self-Supervised Tooth-Level Radiograph Foundations for Faster Dental AI Research and Application.”**

[![DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.17308747-blue.svg)](https://doi.org/10.5281/zenodo.17308747)

Zenodo Link will be publicly available after publication. The model weights and two downstream datasets for peer review are available in the following folder:
https://drive.google.com/drive/folders/1SVCgQTv3M-dGIywCpZ164VPrndHnOwR_?usp=drive_link
        

---

## About this repository
This minimalist repository contains just **two Python files**:
- **`backbone_list.py`** — a compact backbone zoo with the paper’s two main backbones (ViT and ConvNeXt V2) plus popular alternatives, all returning a pooled feature **vector** of shape `[B, C]`.
- **`proj_head.py`** — a projection head (`proj_head`) for self-supervised pretraining and representation learning.

**Pretrained checkpoints (.pth)** are hosted on Zenodo (see DOI above). Each checkpoint bundles **both** the backbone weights and the **matching projection-head** weights.  
Two downstream **demo datasets** are also on Zenodo: `ExAn-MTM Dataset.7z` and `Panoramic Dental Dataset.7z`.

> **Downstream training note (important):**  
> In our downstream experiments, we **fully fine‑tuned** the backbones end‑to‑end on task data. We did **not** freeze the backbone and train only a classifier head for the main results. A linear‑probe baseline can still be reproduced (see optional notes), but it is **not** the configuration used in our study.

---

## Files
### `backbone_list.py`
Backbones included (class names as defined in the file):
- `ViT_backbone` (VisionTransformer)
- `convnextv2_backbone` (ConvNeXt-V2)
- `densenet121_backbone` (DenseNet-121)
- `efficientnetv2_s_backbone` (EfficientNet-V2-S)
- `fastvit_t8_backbone` (FastViT-T8)
- `mambaout_femto_backbone` (MambaOut-Femto)
- `mobilenetv4_backbone` (MobileNet-V4)
- `regnetz_backbone` (RegNet-Z)
- `starnet_s4_backbone` (StarNet-S4)
- `vgg16_backbone` (VGG-16)
- `ghostnetv3_backbone` (GhostNet-V3)

All backbones internally use `timm.create_model(..., features_only=True)`, then apply **global average pooling** and **flatten**, so the forward returns a 2D tensor `[B, C]` suitable for feeding into MLP heads or classifiers.

### `proj_head.py`
- Class: **`proj_head(in_dim, out_dim=256, hidden_dim=256, bottleneck_dim=256, nlayers=3)`**
- Architecture: MLP with GELU and BatchNorm, followed by a WeightNorm linear layer; outputs `[B, out_dim]`.
- **Input requirement:** a 2D tensor `[B, in_dim]` (match `in_dim` to the backbone output channel count `C`).

> **Checkpoint contents (Zenodo)**  
> Each `.pth` contains both the backbone and projection head parameters. The head’s `in_dim` must equal the backbone output size `C`.

---

## Installation
```bash
# (optional) new environment
conda create -n tooth-foundations python=3.10 -y
conda activate tooth-foundations

# core deps
pip install torch torchvision
pip install timm

# (optional) for dataset archives
# Linux: sudo apt-get install p7zip-full
# macOS: brew install p7zip
```

---

## Quick start
> Adjust class names as needed; examples use **ViT** and **ConvNeXt V2** defined in `backbone_list.py`.

### A) Build a backbone and the matching projection head (for SSL / representation learning)
```python
import torch
from backbone_list import ViT_backbone        # or convnextv2_backbone, etc.
from proj_head import proj_head

# 1) Instantiate the backbone (returns pooled [B, C])
backbone = ViT_backbone()

# 2) Infer C from a dummy forward pass
with torch.no_grad():
    feat = backbone(torch.zeros(1, 1, 400, 304))  # ViT validated input size: (H, W) = (400, 304)
    C = feat.shape[1]

# 3) Create the projection head with the matched input dim
head = proj_head(in_dim=C)

# 4) Load a checkpoint that includes backbone & proj_head
ckpt = torch.load("0_vit_combined_ssl.pth", map_location="cpu")

def load_backbone_and_head(backbone, head, sd):
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if "backbone" in sd and "proj_head" in sd:
        backbone.load_state_dict(sd["backbone"], strict=False)
        head.load_state_dict(sd["proj_head"], strict=False)
    else:
        bb = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
        ph = {k.replace("proj_head.", ""): v for k, v in sd.items() if k.startswith("proj_head.")}
        if bb: backbone.load_state_dict(bb, strict=False)
        if ph: head.load_state_dict(ph, strict=False)

load_backbone_and_head(backbone, head, ckpt)
```

### B) Swap a different backbone (e.g., ConvNeXt V2)
```python
from backbone_list import convnextv2_backbone

backbone = convnextv2_backbone()
with torch.no_grad():
    C = backbone(torch.zeros(1, 1, 320, 240)).shape[1]
head = proj_head(in_dim=C)
```

### C) Downstream example (**full fine‑tuning**, as used in our study)
```python
import torch
import torch.nn as nn

num_classes = 2  # example

backbone = ViT_backbone()  # or any backbone from backbone_list.py
with torch.no_grad():
    C = backbone(torch.zeros(1, 1, 400, 304)).shape[1]

# Simple classifier head on top of pooled features
classifier = nn.Linear(C, num_classes)

# Compose end-to-end model (backbone outputs [B, C] -> classifier)
model = nn.Sequential(backbone, classifier)

# Enable gradients for **all** params (full fine-tuning)
for p in model.parameters():
    p.requires_grad = True

# Typical optimizer setup: smaller LR on backbone, larger LR on head
optimizer = torch.optim.AdamW(
    [
        {"params": backbone.parameters(), "lr": 1e-4, "weight_decay": 0.05},
        {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 0.0},
    ]
)

criterion = nn.CrossEntropyLoss()

# minimal training step (pseudo-code)
model.train()
for images, labels in dataloader:
    # images: [B, 1, H, W] radiographs
    logits = model(images)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Resources on Zenodo (DOI: 10.5281/zenodo.17308747)
        
        
### Downstream Datasets
- **ExAn-MTM Dataset.7z** — *41.08 MB*, `md5: ee822b8e231ce570c7d19ca92fef79d7`
- **Panoramic Dental Dataset.7z** — *13.17 MB*, `md5: 4779446046ab2c6ea387e96abd57a9b1`

### Self-supervised checkpoints (include backbone **and** proj_head)
**ViT / ConvNeXt V2 (public, internal, combined):**
- `0_convnextv2_public_ssl.pth` — *112.85 MB*, `md5: 4ba9483039b302bd4e77d16fd64f5f7b`
- `0_convnextv2_combined_ssl.pth` — *112.85 MB*, `md5: 3164bc10d5f6da9cefae2989e58e47d2`
- `0_convnextv2_internal_ssl.pth` — *112.85 MB*, `md5: 07e17eaeb1b57d0661f3dcf90a5a998c`
- `0_vit_combined_ssl.pth` — *86.63 MB*, `md5: 7bbdf511fb66b9c7216e612d379de843`
- `0_vit_public_ssl.pth` — *86.63 MB*, `md5: d37fb4834cf48bba4cd906be5c9c5efb`
- `0_vit_internal_ssl.pth` — *86.63 MB*, `md5: 264dc1eb16abd2563fbd428e0ea3c7a8`

**Other backbones (combined):**
- `1_densenet121_combined.pth` — *29.99 MB*, `md5: bb726a20c38d8c5a2c71fb1c298d35d3`
- `1_efficientnetv2_combined.pth` — *81.07 MB*, `md5: b15fee8caa0fe890208955787d7491f7`
- `1_VGG16_combined.pth` — *59.93 MB*, `md5: e244249c22e806f85e5c8de5d15b3954`
- `1_mobilenetv4_combined.pth` — *122.65 MB*, `md5: 67425bdba2e84b052c903cf1261b2ddd`
- `1_resnext26_combined.pth` — *35.82 MB*, `md5: 4710adb2acc3b1cbe579c6752ad4dc43`
- `1_fastvit_t8_combined.pth` — *13.89 MB*, `md5: 0c72f577215f5b7fe98a17071604c272`
- `1_regnetz_combined.pth` — *108.17 MB*, `md5: c4afcc076aa9819134a72713b62d80ab`
- `1_ghostnetv3_combined.pth` — *9.19 MB*, `md5: 45e4ab4e9a49e47d2726593009e700a4`
- `1_mambaout_combined.pth` — *24.15 MB*, `md5: 9ed069435646374232c2cf80f045df52`
- `1_starnets4_combined.pth` — *29.92 MB*, `md5: 72ba0f77f4eadd682414a2f08bb370de`

> **MD5 verification**
> - Linux: `md5sum <file>`  
> - macOS: `md5 <file>`  
> - Windows (PowerShell): `CertUtil -hashfile <file> MD5`

---

## Notes & tips
- **Input channels:** Backbones in `backbone_list.py` are set up for single‑channel (grayscale) radiographs via `in_chans=1`.
- **Feature shape:** All backbones return `[B, C]` (GAP + flatten). Use that `C` as `in_dim` for `proj_head`.
- **Full fine‑tuning best practices:** Consider a smaller LR for the backbone than the head; use warmup and gradient clipping for stability; mixed precision (`torch.cuda.amp`) is supported.
- **timm models:** Some backbones require specific timm versions; if you hit a name error, try upgrading: `pip install -U timm`.
- **Strict loading:** If you only need the backbone, you can ignore projection head keys by passing `strict=False` in `load_state_dict`.
