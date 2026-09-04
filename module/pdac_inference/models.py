"""GigaPath encoder and Attention MIL — loadable from bundled checkpoints.

Stage-1 weights are self-contained in ``stage1/best_model.pth``.
Stage-2 feature extraction still needs the Prov-GigaPath backbone from
HuggingFace (``prov-gigapath/prov-gigapath``) on first run, then caches it.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import VisionTransformer

ENCODER_EMBED_DIM = 1536
DEFAULT_UNFREEZE_BLOCKS = 4


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _ensure_hf_login(hf_token: Optional[str] = None) -> None:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    except Exception:
        pass


def _hf_access_error(exc: Exception) -> RuntimeError:
    return RuntimeError(
        "Failed to load Prov-GigaPath from HuggingFace.\n"
        "  1) Accept the license: https://huggingface.co/prov-gigapath/prov-gigapath\n"
        "  2) Create a token:     https://huggingface.co/settings/tokens\n"
        "  3) Pass it at runtime:  singularity run --nv --app infer "
        "--env HF_TOKEN=$HF_TOKEN ...\n"
        "  4) Bind a writable cache: -B $WORK/.hf_cache:/opt/pdac/.hf_cache\n"
        f"Original error: {exc}"
    )


def build_gigapath_backbone_from_checkpoint_shapes() -> VisionTransformer:
    """Reconstruct the GigaPath ViT offline (matches Stage-1 checkpoint layout).

    Architecture: ViT-G, patch 16, depth 40, dim 1536, LayerScale, SwiGLUPacked.
    """
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=ENCODER_EMBED_DIM,
        depth=40,
        num_heads=24,
        mlp_ratio=(4096 / 1536) * 2,  # SwiGLUPacked packing
        qkv_bias=True,
        num_classes=0,
        global_pool="",
        act_layer=nn.SiLU,
        mlp_layer=timm.layers.SwiGLUPacked,
        init_values=1e-5,
    )


def build_gigapath_backbone_from_hf(pretrained: bool = True, hf_token: Optional[str] = None):
    """Load official Prov-GigaPath via HuggingFace (used for Stage-2 features)."""
    _ensure_hf_login(hf_token)
    try:
        backbone = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=pretrained,
        )
    except Exception as exc:
        raise _hf_access_error(exc) from exc
    backbone.head = nn.Identity()
    return backbone


class GigaPathClassifier(nn.Module):
    """Prov-GigaPath ViT-g with classification head (Stage 1)."""

    def __init__(
        self,
        num_classes: int = 2,
        head_hidden: int = 512,
        head_dropout: float = 0.3,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else build_gigapath_backbone_from_checkpoint_shapes()
        self.embed_dim = getattr(self.backbone, "embed_dim", ENCODER_EMBED_DIM)
        if hasattr(self.backbone, "head"):
            self.backbone.head = nn.Identity()
        self.head = ClassificationHead(self.embed_dim, num_classes, head_hidden, head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() == 3:
            features = features[:, 0]
        return self.head(features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() == 3:
            features = features[:, 0]
        return features

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int):
        self.freeze_backbone()
        for block in self.backbone.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True


class GigaPathFeatureEncoder(nn.Module):
    """GigaPath backbone for Stage-2 feature extraction (official HF weights)."""

    def __init__(self, hf_token: Optional[str] = None):
        super().__init__()
        self.backbone = build_gigapath_backbone_from_hf(pretrained=True, hf_token=hf_token)
        self.embed_dim = getattr(self.backbone, "embed_dim", ENCODER_EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() == 3:
            features = features[:, 0]
        return features


class AttentionMIL(nn.Module):
    """Attention MIL for slide-level Basal vs Classical."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features, mask=None):
        attn_logits = self.attention(features).squeeze(-1)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn_logits, dim=1)
        bag = torch.bmm(attn.unsqueeze(1), features).squeeze(1)
        return self.classifier(bag), attn


def create_gigapath_model(
    num_classes: int = 2,
    mode: str = "fine_tune",
    unfreeze_blocks: int = DEFAULT_UNFREEZE_BLOCKS,
    hf_token: Optional[str] = None,
) -> GigaPathClassifier:
    del hf_token  # Stage-1 architecture is reconstructed offline
    model = GigaPathClassifier(num_classes=num_classes)
    if mode == "linear_probe":
        model.freeze_backbone()
    elif mode == "fine_tune":
        model.unfreeze_last_n_blocks(unfreeze_blocks)
    elif mode == "full_fine_tune":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model


def load_stage1_checkpoint(path: str, device: torch.device) -> GigaPathClassifier:
    """Load Stage-1 tumor/stroma classifier fully offline from the bundled .pth."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mode = ckpt.get("mode", "fine_tune")
    model = create_gigapath_model(num_classes=2, mode=mode)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Stage-1 load mismatch: missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    return model


def load_stage2_checkpoint(path: str, device: torch.device) -> tuple[AttentionMIL, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    feat_dim = ckpt.get("feat_dim", ENCODER_EMBED_DIM)
    mil_hidden = ckpt.get("mil_hidden", 256)
    mil = AttentionMIL(input_dim=feat_dim, hidden_dim=mil_hidden, num_classes=2)
    mil.load_state_dict(ckpt["model_state_dict"])
    mil.to(device).eval()
    return mil, ckpt
