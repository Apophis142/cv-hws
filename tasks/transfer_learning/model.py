from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def create_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def configure_stage(model, fine_tune: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)
    if fine_tune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad_(True)


def training_mode(model, fine_tune: bool) -> None:
    model.eval()
    model.fc.train()
    if fine_tune:
        model.layer4.train()
