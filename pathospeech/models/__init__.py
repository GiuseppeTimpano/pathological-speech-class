"""Model construction for the proposed CNN-BiGRU and the pretrained baselines."""

from pathlib import Path

from safetensors.torch import load_file
from transformers import ASTForAudioClassification, AutoModelForAudioClassification

from pathospeech.config import ExperimentConfig
from pathospeech.models.cnn_bigru import CNNBiGRUClassifier, CNNGRUConfig


def cnn_bigru_config(cfg: ExperimentConfig, num_labels: int) -> CNNGRUConfig:
    return CNNGRUConfig(
        input_dim=cfg.n_mels,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_classes=num_labels,
        eca_attention=cfg.eca,
        temporal_attention=cfg.tam,
    )


def build_model(cfg: ExperimentConfig, label2id: dict, id2label: dict):
    """Fresh CNN-BiGRU, or a pretrained Wav2Vec2 / HuBERT / AST with a new classification head."""
    num_labels = len(id2label)
    if cfg.is_cnn_bigru:
        return CNNBiGRUClassifier(cnn_bigru_config(cfg, num_labels))
    model_cls = ASTForAudioClassification if cfg.is_ast else AutoModelForAudioClassification
    return model_cls.from_pretrained(
        cfg.pretrained_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=cfg.is_ast,  # AST ships with a 527-class AudioSet head
    )


def load_trained_model(cfg: ExperimentConfig, model_dir: Path, num_labels: int):
    """Load a fold's best model saved by train.py."""
    model_dir = Path(model_dir)
    if cfg.is_cnn_bigru:
        model = CNNBiGRUClassifier(cnn_bigru_config(cfg, num_labels))
        model.load_state_dict(load_file(str(model_dir / "model.safetensors")))
        return model
    model_cls = ASTForAudioClassification if cfg.is_ast else AutoModelForAudioClassification
    return model_cls.from_pretrained(model_dir)


__all__ = ["CNNBiGRUClassifier", "CNNGRUConfig", "build_model", "load_trained_model", "cnn_bigru_config"]
