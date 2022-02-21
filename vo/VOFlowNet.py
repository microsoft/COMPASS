import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if bn_layer:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else: 
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True)
        )

def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), 
        nn.ReLU(inplace=True)
        )

def feature_extract(bn_layer, intrinsic):

    if intrinsic:
        inputnum = 4
    else:
        inputnum = 2
    conv1 = conv(inputnum, 16, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 320 x 240
    conv2 = conv(16,32, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 160 x 120
    conv3 = conv(32,64, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 80 x 60
    conv4 = conv(64,128, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 40 x 30
    conv5 = conv(128,256, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 20 x 15

    conv6 = conv(256,512, kernel_size=5, stride=5, padding=0, bn_layer=bn_layer) # 4 x 3
    conv7 = conv(512,1024, kernel_size=(3, 4), stride=1, padding=0, bn_layer=bn_layer) # 1 x 1

    return nn.Sequential(conv1, conv2, conv3, conv4,
                                conv5, conv6, conv7,  
        )

class VOFlowRes(nn.Module):
    def __init__(self, intrinsic=False, down_scale=False, fcnum = 512):
        super(VOFlowRes, self).__init__()

        fc1_trans = linear(fcnum, 128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(fcnum, 128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)


        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)


    def forward(self, x):
        #TODO 
        # load pretraind encoder preEnc 
        # encode feature z from preEnc
             
        x = x.view(x.shape[0], -1)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)
        return torch.cat((x_trans, x_rot), dim=1)

if __name__ == '__main__':
    
    voflownet = VOFlowRes(down_scale=True, config=1) # 
    voflownet.cuda()
    print(voflownet)
    import numpy as np
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=4, threshold=10000)
    imsize1 = 112
    imsize2 = 160
    x, y = np.ogrid[:imsize1, :imsize2]
    # print x, y, (x+y)
    img = np.repeat((x + y)[..., np.newaxis], 2, 2) / float(imsize1 + imsize2)
    img = img.astype(np.float32)
    print(img.dtype, img.shape)
    # print img

    imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    imgTensor = torch.from_numpy(imgInput)
    print(imgTensor.shape)
    z = voflownet(imgTensor.cuda())
    print(z.data.shape)
    print(z.data.cpu().numpy())

    # for name,param in voflownet.named_parameters():
    #   print name,param.requires_grad


