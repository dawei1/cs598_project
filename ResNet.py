'''
This module defines the ResNet model that forms the first part of the overall
model architecture. It uses a pre-trained ResNet50 model.
'''

def get_resnet_model():
    import torchvision.models
    import torch.nn
    model = torchvision.models.resnet50(pretrained=True)

    '''
    for param in model.named_parameters():
        print(param[0])
        if param[0] == 'conv1.weight':
            print('Keep the first conv2d layer trainable')
            continue
        if param[0] == 'fc.weight':
            print('Keep the last fc layer trainable')
            continue
        if param[0] == 'fc.bias':
            print('Keep the last fc layer trainable')
            continue
        # Freeze the pretrained model layers
        #param[1].requires_grad = False
    '''
    model_list = list(model.children())[:-2]
    model = torch.nn.Sequential(*model_list)
    return model
