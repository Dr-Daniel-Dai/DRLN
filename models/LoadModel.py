import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels

from config import pretrained_model

import pdb

class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        print(self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = models.densenet121(pretrained=True)
            # self.model = getattr(models, self.backbone_arch)()
            # if self.backbone_arch in pretrained_model:
            #     self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
            #     print ('data loaded')
        # else:
        #     if self.backbone_arch in pretrained_model:
        #         self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)
        #     else:
        #         self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000)
        
        out_chann = 512
        
        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])     ###   input 512x512 so output 2048x14x14 batch channel hight width
        if self.backbone_arch == 'densenet121' or self.backbone_arch == 'se_densenet121':
            self.model = nn.Sequential(*list(self.model.features.children())[:-2])
            print ("densenet loaded")
        if self.backbone_arch == 'vgg16' or self.backbone_arch == 'se_vgg16':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool = nn.AdaptiveMaxPool2d(output_size=1)
        
        self.classifier = nn.Linear(out_chann*2, self.num_classes, bias=False)
        self.classifier_classi = nn.Linear(out_chann, 7, bias=False)
        self.cal_sigmoid = nn.Sigmoid()
        
        if self.use_dcl:
            if config.cls_2:
                self.classifier_swap = nn.Linear(out_chann, 2, bias=False)
                
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(out_chann, 2*self.num_classes, bias=False)
                
            ####dy   nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
            self.Convmask = nn.Conv2d(out_chann, 1, 1, stride=1, padding=0, bias=True)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)

        if self.use_Asoftmax:
            # self.Aclassifier = AngleLinear(2048, self.num_classes, bias=False)
            self.Aclassifier = nn.Linear(out_chann, self.num_classes, bias=False)
            
    def forward(self, x, last_cont=None):
        x = self.model(x)       #####  2048x14x14
        # print (x.size())      #####  521x14x14
        if self.use_dcl:
            mask = self.Convmask(x)    ###dy  14x14x 1 hotmap 
            mask = self.avgpool2(mask)   ## 7x7
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)    ###  batchsize x 49

            
        # if self.use_dcl:
        #     # mask = self.Convmask(x)    ###dy  14x14x 1 hotmap   index
        #     mask = self.avgpool2(x)   ## 7x7x512
        #     mask = self.cal_sigmoid(mask)
        #     # mask = mask.view(mask.size(0), -1)    ###  batchsize x 49
            
        
        x = self.avgpool(x)   ### dy  1x1x2048
        x = x.view(x.size(0), -1)    ##dy   batchsize x 2048 
        
        idx_unswap = torch.tensor([0,2,4,6,8],dtype=torch.long).cuda()
        idx_swap = torch.tensor([1,3,5,7,9],dtype=torch.long).cuda()
        
        unswap_select = torch.index_select(x,dim = 0, index= idx_unswap)
        swap_select = torch.index_select(x,dim = 0, index= idx_swap)

        x_cat = torch.cat([unswap_select,swap_select],1)    ###  batch x  512+512
        
        out = []
        out_cla = self.classifier(x_cat)
        out_cla_sig = self.cal_sigmoid(out_cla)
        
        
        out.append(out_cla_sig)     ###dy   out[0] batchsize x numclass
        # print (self.classifier(x).size())      #####  12
        
        if self.use_dcl:
            out.append(self.classifier_swap(x))    ###dy out[1]  batch x 2     adverisal loss
            out.append(mask)                       ###dy  out[2] batch x 49

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))
                
        # out.append(self.classifier_classi(x))     ###dy  out[3] batch x 7   CE loss
        return out
