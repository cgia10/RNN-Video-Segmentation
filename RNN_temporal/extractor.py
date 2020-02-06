import torch
import torchvision.models as models

class Feat_extractor(torch.nn.Module):

    def __init__(self):
        super(Feat_extractor, self).__init__()  
        resnet50 = models.resnet50(pretrained=True)
        modules = list(resnet50.children())[:-1]
        resnet50 = torch.nn.Sequential(*modules)
        self.resnet50 = resnet50 # Nx3x240x320 -> Nx2048x2x4    

    def forward(self, img):
        x = self.resnet50(img)
        return x