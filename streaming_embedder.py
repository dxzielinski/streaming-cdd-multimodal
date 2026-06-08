from __future__ import annotations

from typing import Optional
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import HashingVectorizer
import torch
from torch import nn


SUPPORTED_MODALITIES = ("image", "text", "both")


class _ResNet18EmbeddingModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x


def _simple_image_embedding(
    path: str, thumb_size: int = 8, hist_bins: int = 16
) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    thumb = img.resize((thumb_size, thumb_size))
    thumb_arr = np.asarray(thumb, dtype=np.float32).reshape(-1) / 255.0
    hists = []
    for c in range(3):
        hist, _ = np.histogram(
            arr[..., c], bins=hist_bins, range=(0.0, 1.0), density=True
        )
        hists.append(hist.astype(np.float32))
    hist_feat = np.concatenate(hists, axis=0)
    return np.concatenate([means, stds, thumb_arr, hist_feat], axis=0).astype(
        np.float32
    )


def _unit_normalize(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm < 1e-12:
        return x.astype(np.float32, copy=False)
    return (x / norm).astype(np.float32, copy=False)


class StreamingEmbedder:
    """Compute multimodal embeddings one sample at a time as they arrive.

    Modalities:
      - "image": ResNet18 features only.
      - "text":  HashingVectorizer features only.
      - "both":  L2-normalized concat of (weighted) image and text features.
    """

    def __init__(
        self,
        modality: str,
        *,
        text_features: int = 256,
        image_weight: float = 1.0,
        text_weight: float = 1.0,
        device: str = "cpu",
    ):
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality: {modality}. Choose one of {SUPPORTED_MODALITIES}."
            )
        self.modality = modality
        self.text_features = int(text_features)
        self.image_weight = float(image_weight)
        self.text_weight = float(text_weight)
        self.device = device

        self._image_model: Optional[_ResNet18EmbeddingModel] = None
        self._image_transform = None
        self._image_fallback = False
        self._text_vectorizer: Optional[HashingVectorizer] = None

        if modality in ("image", "both"):
            self._init_image_model()
        if modality in ("text", "both"):
            self._text_vectorizer = HashingVectorizer(
                n_features=self.text_features, alternate_sign=False, norm=None
            )

    def _init_image_model(self) -> None:
        if torch is None:
            self._image_fallback = True
            return
        try:
            from torchvision import models, transforms

            self._image_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self._image_model = _ResNet18EmbeddingModel(models).to(self.device)
            self._image_model.eval()
            self._image_fallback = False
        except Exception:
            self._image_fallback = True

    @property
    def using_image_fallback(self) -> bool:
        return self._image_fallback

    def _embed_image(self, path: str) -> np.ndarray:
        if self._image_fallback or self._image_model is None:
            return _simple_image_embedding(path)
        img = Image.open(path).convert("RGB")
        x = self._image_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._image_model(x).cpu().numpy().ravel()
        return emb.astype(np.float32, copy=False)

    def _embed_text(self, text: str) -> np.ndarray:
        assert self._text_vectorizer is not None
        X = self._text_vectorizer.transform([text])
        return X.toarray().astype(np.float32).ravel()

    def embed(self, *, image_path: Optional[str], text: Optional[str]) -> np.ndarray:
        if self.modality == "image":
            if image_path is None:
                raise ValueError("image_path is required for modality='image'")
            return _unit_normalize(self._embed_image(image_path))
        if self.modality == "text":
            if text is None:
                raise ValueError("text is required for modality='text'")
            return _unit_normalize(self._embed_text(text))
        if image_path is None or text is None:
            raise ValueError(
                "image_path and text are both required for modality='both'"
            )
        img = _unit_normalize(self._embed_image(image_path)) * self.image_weight
        txt = _unit_normalize(self._embed_text(text)) * self.text_weight
        return _unit_normalize(np.concatenate([img, txt], axis=0))
