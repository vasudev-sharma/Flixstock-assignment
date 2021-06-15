import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


model_dict = {
        # DenseNet
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,

        # ResNet
        'resnet50': 2048,
        'resnet101' : 2048,
        'resnet34': 512,

        # EfficientNet
        'efficientnet-b0':1280,
        'efficientnet-b3': 1536,
        'efficientnet-b5': 2048

}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(model_name, n_labels=21, use_pretrained=False):

    if not model_name.startswith('efficientnet'):
        model = torch.hub.load('pytorch/vision:v0.9.0', model_name, pretrained=use_pretrained)
    else:
        model = EfficientNet.from_pretrained(model_name)

    # Densenet 
    if model_name.startswith('densenet'): 
        model.classifier = nn.Linear(model_dict[model_name], n_labels)

    if model_name.startswith('efficientnet'):
        print(model_dict[model_name])
        model._fc = nn.Linear(model_dict[model_name], n_labels)
    else:
        model.fc = nn.Linear(model_dict[model_name], n_labels)

    # Migrate the mode to device
    model.to(device)
    return model




if __name__ == 'main':

    model_dict = {
        # DenseNet
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,

        # ResNet
        'resnet50': 2048,
        'resnet101' : 2048,
        'resnet34': 512,

        # EfficientNet
        'efficientnet-b0':1280,
        'efficientnet-b3': 1536,
        'efficientnet-b5': 2048

    }

    # Change the model name depending which model you want to fine tune
    model_name = 'efficientnet-b3'


