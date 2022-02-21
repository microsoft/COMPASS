# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
#from pfm import readPFM
from utils import bilinear_interpolate
from FLO import read_flo
from Datasets.DataLoaderBase import DataLoaderBase

# for sintel and flyingchairs
class SintelFlowDataset(DataLoaderBase):

    def make_flow_mask_filename(self, fn):
        """fn is the flow file name. """
        return None

    def load_flow(self, fn, maskFn=None):
        """This function should return 2 objects, flow and mask. """
        return read_flo(fn), None

    def load_image(self, fn):
        return cv2.imread(fn)

    def load_motion(self, idx):
        return None


class FlyingFlowDataset(DataLoaderBase):

    def make_flow_mask_filename(self, fn):
        """fn is the flow file name. """
        return None

    def load_flow(self, fn, maskFn=None):
        """This function should return 2 objects, flow and mask. """
        flow, scale0 = readPFM(fn)
        return flow[:,:,:2], flow[:,:,2]

    def load_image(self, fn):
        return cv2.imread(fn)

    def load_motion(self, idx):
        return None

# CROSS_OCC = 1
# SELF_OCC  = 10
# OUT_OF_FOV = 100
class TartanFlowDataset(DataLoaderBase):

    def make_flow_mask_filename(self, fn):
        """fn is the flow file name. """
        return fn.replace('flow.npy', 'mask.npy')

    def load_flow(self, fn, maskFn=None):
        """This function should return 2 objects, flow and mask. """
        flow = np.load(fn)
        if self.flagFlowMask and maskFn is not None:
            mask = np.load(maskFn)
            mask_cross = np.zeros_like(mask).astype(np.uint8) # 0 - valid, 255 - cross occlusion
            mask_cross[mask == 1] = 255
        else:
            mask_cross = None
        return flow, mask_cross

    def load_image(self, fn):
        return cv2.imread(fn)

    def load_motion(self, idx):
        if ( self.motions is not None ):
            return self.motions[idx]
        else:
            return None

if __name__ == '__main__':

    import time
    from utils import visflow, RandomResizeCrop, RandomHSV, ToTensor, Normalize, tensor2img, Compose, FlipFlow
    from data_roots import *

    normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    randResizeCrop =  RandomResizeCrop(size=(400, 400), max_focal=320.0, keep_center=True, fix_ratio=False)
    transformlist = [ randResizeCrop, RandomHSV((10,80,80), random_random=0.5), FlipFlow(), ToTensor(), normalize ] # 

    datafile = 'data/hm01_flow.txt'

    # dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
    # dataroot_list = FLOW_DR[dataset_term]['local']

    flowDataset = TartanFlowDataset(datafile, 
                                    imgCPrefix='', flowCPrefix='',
                                    transform=Compose(transformlist), has_mask=False)
    
    for k in range(0,len(flowDataset),1):
        sample = flowDataset[k]
        flownp = sample['flow'].numpy()
        flownp = flownp / flowDataset.flow_norm

        # flowmask = sample['fmask'].squeeze()

        # import ipdb;ipdb.set_trace()
        flownp = flownp.transpose(1,2,0)
        flowvis = visflow(flownp)
        # flowvis[flowmask>128,:] = 0
        img1 = sample['img1']
        img2 = sample['img2']
        img1 = tensor2img(img1, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        img2 = tensor2img(img2, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        imgdisp = np.concatenate((img1, img2 ,flowvis), axis=0)
        imgdisp = cv2.resize(imgdisp,(0,0),fx=0.5,fy=0.5)
        cv2.imshow('img',imgdisp)
        cv2.waitKey(0)

        # # warp image2 to image1
        # warp21 = np.zeros_like(img1)
        # warp12 = np.zeros_like(img1)
        # for h in range(img1.shape[0]):
        #     for w in range(img1.shape[1]):
        #         th = h + flownp[h,w,1] 
        #         tw = w + flownp[h,w,0] 
        #         if round(th)>=0 and round(th)<img1.shape[0] and round(tw)>=0 and round(tw)<img1.shape[1]:
        #             warp21[h,w,:] = bilinear_interpolate(img2, th, tw).astype(np.uint8)
        #             warp12[int(round(th)), int(round(tw)), :] = img1[h,w,:]

        # # # import ipdb;ipdb.set_trace()
        # diff = warp21.astype(np.float32) - img1.astype(np.float32)
        # diff = np.abs(diff)
        # print 'diff:', diff.mean()
        # diff = diff.astype(np.uint8)
        # con1 = np.concatenate((img1,warp21,flowvis),axis=0)
        # con2 = np.concatenate((img2, warp12,diff), axis=0)
        # imgdisp = np.concatenate((con1, con2), axis=1)
        # imgdisp = cv2.resize(imgdisp,(0,0),fx=0.5,fy=0.5)
        # cv2.imshow('img',imgdisp)
        # cv2.waitKey(0)
