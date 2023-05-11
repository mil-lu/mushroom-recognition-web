import torch
import torch.nn as nn
import torchvision.models as models
from timm.models import vit_base_patch16_224

class CombinedModel(nn.Module):
    def __init__(self, num_classes=9):
        super(CombinedModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = True
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        vit = vit_base_patch16_224(pretrained=True)
        vit.head = nn.Linear(vit.head.in_features, num_classes)

        self.resnet = resnet
        self.vit = vit
        self.classifier = nn.Linear(resnet.fc.out_features + vit.head.out_features, num_classes)

    def forward(self, x):
        resnet_features = self.resnet(x)
        vit_features = self.vit(x)
        combined_features = torch.cat((resnet_features, vit_features), dim=1)
        output = self.classifier(combined_features)
        return output
