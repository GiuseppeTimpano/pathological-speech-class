"""CNN-BiGRU classifier with optional ECA and temporal attention.

Parameter names match the checkpoints saved by ``train.py``, so ``model.safetensors`` files
load without any key remapping.
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from pathospeech.models.attention import ECABlock2D, TemporalAttention


class CNNGRUConfig(PretrainedConfig):
    def __init__(self, input_dim: int = 128, hidden_size: int = 128, num_layers: int = 2, num_classes: int = None,
                 eca_attention: bool = False, temporal_attention: bool = False, **kwargs):
        eca = kwargs.pop("eca", eca_attention)  # "eca" is the key stored in config.json
        kwargs.pop("cbam", None)                 # key present in older checkpoints, never enabled
        super().__init__(**kwargs)
        self.input_dim = input_dim               # number of Mel bands
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if num_classes is not None:
            self.num_labels = num_classes
        self.eca = eca
        self.temporal_attention = temporal_attention


class CNNBiGRUClassifier(PreTrainedModel):
    """log-Mel (B, T, F) -> 3 VGG-style conv blocks -> [ECA] -> 2-layer BiGRU -> [TAM | mean] -> linear."""

    config_class = CNNGRUConfig

    def __init__(self, config: CNNGRUConfig):
        super().__init__(config)
        self.config = config

        def conv_block(c_in, c_out):
            return [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
            ]

        # three blocks, each halving time and frequency: (B, 128, T/8, F/8)
        self.conv = nn.Sequential(*conv_block(1, 32), *conv_block(32, 64), *conv_block(64, 128))

        if config.eca:
            self.eca_block = ECABlock2D(kernel_size=5)

        self.gru = nn.GRU(
            input_size=128 * (config.input_dim // 8),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        if config.temporal_attention:
            self.attention_layer = TemporalAttention(config.hidden_size)

        self.dropout = nn.Dropout(0.5)
        self.layernorm = nn.LayerNorm(config.hidden_size * 2)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_labels)

    def forward(self, input_values: torch.Tensor = None, labels: torch.Tensor = None, **kwargs):
        x = self.conv(input_values.unsqueeze(1))           # (B, 128, T', F')
        if self.config.eca:
            x = self.eca_block(x)

        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)  # one feature vector per time step

        out, _ = self.gru(x)                                # (B, T', 2H)
        out = self.layernorm(out)
        out = self.attention_layer(out) if self.config.temporal_attention else out.mean(dim=1)
        logits = self.fc(self.dropout(out))

        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


# name used in the config.json of the saved checkpoints
CNNGRUClassifier = CNNBiGRUClassifier
