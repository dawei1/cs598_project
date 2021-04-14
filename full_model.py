import torch

import Patching as p
import Recognition as r


class PatchingModel(torch.nn.Module):

    def __init__(self, height_width, P, c_prime):
        super(PatchingModel, self).__init__()
        self.patching = p.Patching(height_width, P)
        self.recognition = r.Recognition(c_prime)

    def forward(self, x):

        output = self.patching(output)
        output = self.recognition(output)
        return output
