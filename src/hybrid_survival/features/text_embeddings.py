"""Clinical text embeddings via HuggingFace transformers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    pass

warnings.filterwarnings("ignore")


class ClinicalBERTEmbedder:
    """Encode clinical notes with a biomedical BERT checkpoint (default: Bio_ClinicalBERT)."""

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Loading encoder: {model_name} | device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def encode_corpus(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Encoding notes")
        for i in iterator:
            embeddings.append(self.encode_batch(texts[i : i + self.batch_size]))
        return np.vstack(embeddings)


class TextPreprocessor:
    def __init__(self, lowercase: bool = True, remove_special_chars: bool = False, max_length: int = 5000):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.max_length = max_length

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text[: self.max_length]
        if self.lowercase:
            text = text.lower()
        if self.remove_special_chars:
            import re

            text = re.sub(r"[^a-zA-Z0-9\s\.]", "", text)
        text = " ".join(text.split())
        return text

    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        return [self.preprocess(t) for t in texts]


def extract_text_features(
    clinical_notes: List[str],
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_length: int = 512,
    batch_size: int = 16,
    preprocess: bool = True,
    embedder: Optional[ClinicalBERTEmbedder] = None,
) -> np.ndarray:
    if preprocess:
        clinical_notes = TextPreprocessor().preprocess_corpus(clinical_notes)
    enc = embedder or ClinicalBERTEmbedder(model_name=model_name, max_length=max_length, batch_size=batch_size)
    return enc.encode_corpus(clinical_notes)
