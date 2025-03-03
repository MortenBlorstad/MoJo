
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


config = {
    "encoder": {
        "input_size": 3,
        "hidden_size": 256,
        "num_layers": 1,
        "dropout": 0.1,
    },

    "decoder": {
        "output_size": 3,
        "hidden_size": 256,
        "num_layers": 1,
        "dropout": 0.1,
    }
}

class WorldModel(nn.Module):
    def __init__(self, config):
        super(WorldModel, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.encoder(inputs)
        x = self.core(x)
        x = self.decoder(x)
        return x

    def get_loss(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.decoder.get_loss(inputs, outputs)

    def get_metrics(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.decoder.get_metrics(inputs, outputs)

    def get_metrics_names(self) -> List[str]:
        return self.decoder.get_metrics_names()

