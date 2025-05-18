import torch
from torch import nn
from torchvision import models

class MobileNetV2Model(nn.Module):
    def __init__(self, output_shape, freeze_extractor_parameters = True):
        super().__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        if freeze_extractor_parameters:
            for param in self.model.parameters(): # freezing the existing weights of the model for faster training
                param.requires_grad = False
        
        classif_in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=classif_in_features, out_features=output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class TinyVGGModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=4,
                      padding=0,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=4,
                      padding=0,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=4,
                      padding=0,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=4,
                      padding=0,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*11*11,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv_block_1(x)
        result = self.conv_block_2(result)
        result = self.classifier(result)
        return result
    
output_shape = 38