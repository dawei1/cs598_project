import torch


class Patching(torch.nn.Module):

    def __init__(self, height_width, P):
        super(Patching, self).__init__()
        if P == height_width:
            self.patching = lambda x: x
        elif P > height_width:
            self.patching = torch.nn.Upsample(size=(P, P))
        else:  # P < height_width
            kernel_size = height_width - P + 1
            self.patching = torch.nn.MaxPool2d(kernel_size, stride=1)

    def forward(self, x):
        return self.patching(x)
