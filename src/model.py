import torchvision.models as models
import torch.nn as nn

# def build_model(pretrained=True, fine_tune=False, num_classes=10):
#     if pretrained:
#         print('[INFO]: Loading pre-trained weights')
#     else:
#         print('[INFO]: Not loading pre-trained weights')
#     model = models.mobilenet_v3_large(pretrained=pretrained)
#     if fine_tune:
#         print('[INFO]: Fine-tuning all layers...')
#         for params in model.parameters():
#             params.requires_grad = True
#     elif not fine_tune:
#         print('[INFO]: Freezing hidden layers...')
#         for params in model.parameters():
#             params.requires_grad = False
#     # Change the final classification head.
#     model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)
#     return model


def build_model(pretrained=True, fine_tune=False, num_classes=3):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    
    model = models.mobilenet_v2(pretrained=pretrained)
    
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    else:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    
    # Change the final classification head.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)
    
    return model