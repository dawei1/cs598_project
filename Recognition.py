import torch
import torch.nn.functional as F

class Recognition(torch.nn.Module):
    #PxPxc_cprime output dimensions of patch slicing
    #k is number of class - in our case likely 2
    def __init__(self, c_prime):

        C_STAR = 512 #Paper says set C* to 512 
        NUM_CLASSES = 14 #14 different diseases
        super(Recognition, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_size = c_prime, output_size = C_STAR,
                                     kernel_size=(3,3), stride = 1)

        self.batch_norm = torch.nn.BatchNorm2d(num_features=NUM_CLASSES)
        #cite paper with torch.nn batch norm         
        self.conv2 = torch.nn.Conv2d(input_size = C_STAR, output_size = NUM_CLASSES, 
                                     kernel_size=(1,1), stride = 1)

        self.sig = torch.nn.Sigmoid()


    def forward(self, x):
        #first conv
        patch_features = self.conv1(x)
        #batch normalization
        patch_features = self.batch_norm(patch_features)
        #relu
        patch_features = F.relu(patch_features)
        #through a 1x1 conv layer to generate set of PxP
        patch_scores = self.conv2(patch_features)
        #normalized by logistic Sigmoid function
        patch_scores = self.sig(patch_scores)
        #final predictions with K channels
        return patch_scores
                
