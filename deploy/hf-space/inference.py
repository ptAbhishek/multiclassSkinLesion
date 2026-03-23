"""Inference utilities for the Hugging Face Space app."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

CLASS_NAMES = ["nv", "mel", "bkl", "bcc", "akiec", "df", "vasc"]
IMG_SIZE = 224

# Keep deployment self-contained without requiring a global install.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
_LOCAL_PACKAGE_SRC = _PROJECT_ROOT / "ibr_pypi_module" / "src"
if str(_LOCAL_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PACKAGE_SRC))

from ibr_defused_algo.tensorflow_models import (
    ConvStage,
    DefusedIBR5IBR6,
    IBR5Net,
    IBR6Net,
    IBRBase,
)


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Set it in Hugging Face Space Variables."
        )
    return value


@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    """Load and cache model from HF Hub."""
    repo_id = _get_required_env("MODEL_REPO_ID")
    filename = os.getenv("MODEL_FILENAME", "best_model.keras").strip() or "best_model.keras"
    token = os.getenv("HF_TOKEN", "").strip() or None

    model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    custom_objects = {
        "ConvStage": ConvStage,
        "IBRBase": IBRBase,
        "IBR5Net": IBR5Net,
        "IBR6Net": IBR6Net,
        "DefusedIBR5IBR6": DefusedIBR5IBR6,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    # Warmup avoids first-request overhead spikes on CPU Spaces.
    _ = model(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32), training=False)
    return model


def preprocess_image(image: Image.Image) -> tf.Tensor:
    if image is None:
        raise ValueError("No image provided. Upload a valid skin lesion image.")

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return tf.convert_to_tensor(arr, dtype=tf.float32)


def predict(image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
    model = load_model()
    x = preprocess_image(image)

    logits = model(x, training=False)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_confidence = float(probs[top_idx])

    score_map = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}
    return top_class, top_confidence, score_map
