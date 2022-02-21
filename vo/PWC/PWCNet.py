"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from .correlation import FunctionCorrelation
import cv2 # debug

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)



class PWCDCNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, md=4, flow_norm=20.0):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNet,self).__init__()

        self.flow_norm = flow_norm
        
        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128,128, kernel_size=3, stride=1)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.conv6aa = conv(128,196, kernel_size=3, stride=2)
        self.conv6a  = conv(196,196, kernel_size=3, stride=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        # self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4]) 
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4]) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4]) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.ones(x.size()).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask


    def forward(self,x):
        im1 = x[0]
        im2 = x[1]
        
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))


        # corr6 = self.corr(c16, c26) 
        corr6 = FunctionCorrelation(tenFirst=c16, tenSecond=c26)

        corr6 = self.leakyRELU(corr6)   


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        
        warp5 = self.warp(c25, up_flow6*0.625)
        # corr5 = self.corr(c15, warp5) 
        corr5 = FunctionCorrelation(tenFirst=c15, tenSecond=warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24, up_flow5*1.25) #1.25
        # corr4 = self.corr(c14, warp4)  
        corr4 = FunctionCorrelation(tenFirst=c14, tenSecond=warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23, up_flow4*2.5)#2.5
        # corr3 = self.corr(c13, warp3) 
        corr3 = FunctionCorrelation(tenFirst=c13, tenSecond=warp3)
        corr3 = self.leakyRELU(corr3)
        

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)


        warp2 = self.warp(c22, up_flow3*5.0) 
        # corr2 = self.corr(c12, warp2)
        corr2 = FunctionCorrelation(tenFirst=c12, tenSecond=warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
 
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        
        # if self.training:
        return flow2,flow3,flow4,flow5,flow6
        # else:
        #     return flow2

    def scale_targetflow(self, targetflow, small_scale=False):
        '''
        calculte GT flow in different scales 
        '''
        if small_scale:
            target4 = targetflow
        else:
            target4 = F.interpolate(targetflow, scale_factor=0.25, mode='bilinear', align_corners=True) #/4.0
        target8 = F.interpolate(target4, scale_factor=0.5, mode='bilinear', align_corners=True) #/2.0
        target16 = F.interpolate(target8, scale_factor=0.5, mode='bilinear', align_corners=True) #/2.0
        target32 = F.interpolate(target16, scale_factor=0.5, mode='bilinear', align_corners=True) #/2.0
        target64 = F.interpolate(target32, scale_factor=0.5, mode='bilinear', align_corners=True) #/2.0
        return (target4, target8, target16, target32, target64)

    def scale_mask(self, mask, threshold=128, small_scale=False):
        '''
        in tarranair, 
        mask=0:   Valid
        mask=1:   CROSS_OCC
        mask=10:  SELF_OCC
        mask=100: OUT_OF_FOV
        '''
        if small_scale:
            mask4 = mask
        else:
            mask4 = F.interpolate(mask, scale_factor=0.25, mode='bilinear', align_corners=True) #/4.0
        mask8 = F.interpolate(mask4, scale_factor=0.5, mode='bilinear', align_corners=True) #/2.0
        mask16 = F.interpolate(mask8, scale_factor=0.5, mode='bilinear', align_corners=True) #/2.0
        mask32 = F.interpolate(mask16, scale_factor=0.5, mode='bilinear', align_corners=True) #/2.0
        mask64 = F.interpolate(mask32, scale_factor=0.5, mode='bilinear', align_corners=True) #/2.0
        mask4 = mask4<threshold
        mask8 = mask8<threshold
        mask16 = mask16<threshold
        mask32 = mask32<threshold
        mask64 = mask64<threshold
        return (mask4, mask8, mask16, mask32, mask64)

    def get_loss(self, output, target, criterion, small_scale=False):
        '''
        return flow loss
        '''
        if self.training:
            target4, target8, target16, target32, target64 = self.scale_targetflow(target, small_scale)
            loss1 = criterion(output[0], target4)
            loss2 = criterion(output[1], target8)
            loss3 = criterion(output[2], target16)
            loss4 = criterion(output[3], target32)
            loss5 = criterion(output[4], target64)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5.0
        else:
            if small_scale:
                output4 = output[0]
            else:
                output4 = F.interpolate(output[0], scale_factor=4, mode='bilinear', align_corners=True)# /4.0
            loss = criterion(output4, target)
        return loss


    def get_loss_w_mask(self, output, target, criterion, mask, small_scale=False):
        '''
        return flow loss
        small_scale: True - the target and mask are of the same size with output
                     False - the target and mask are of 4 time size of the output
        '''
        if self.training: # PWCNet + training
            target4, target8, target16, target32, target64 = self.scale_targetflow(target, small_scale)
            mask4, mask8, mask16, mask32, mask64 = self.scale_mask(mask, small_scale=small_scale) # only consider coss occlution which indicates moving objects
            mask4 = mask4.expand(target4.shape)
            mask8 = mask8.expand(target8.shape)
            mask16 = mask16.expand(target16.shape)
            mask32 = mask32.expand(target32.shape)
            mask64 = mask64.expand(target64.shape)
            loss1 = criterion(output[0][mask4], target4[mask4])
            loss2 = criterion(output[1][mask8], target8[mask8])
            loss3 = criterion(output[2][mask16], target16[mask16])
            loss4 = criterion(output[3][mask32], target32[mask32])
            loss5 = criterion(output[4][mask64], target64[mask64])
            loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5.0
        else:
            if small_scale:
                output4 = output[0]
            else:
                output4 = F.interpolate(output[0], scale_factor=4, mode='bilinear', align_corners=True)# /4.0
            valid_mask = mask < 10
            valid_mask = valid_mask.expand(target.shape)
            loss = criterion(output4[valid_mask], target[valid_mask])
        return loss

def pwc_dc_net(path=None):

    model = PWCDCNet()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model



if __name__ == '__main__':
    
    flownet = PWCDCNet()
    flownet.cuda()
    print(flownet)
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    np.set_printoptions(precision=4, threshold=100000)
    image_width = 512
    image_height = 384
    x, y = np.ogrid[:image_width, :image_height]
    # print x, y, (x+y)
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(image_width + image_height)
    img = img.astype(np.float32)
    print(img.dtype)

    imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    imgTensor = torch.from_numpy(imgInput)
    start_time = time.time()
    # for k in range(100):
    import ipdb;ipdb.set_trace()
    z = flownet([imgTensor.cuda(),imgTensor.cuda()])
    # print z[0].data.cpu().numpy().shape
    print(z[0].data.cpu().numpy())
        # print time.time() - start_time

