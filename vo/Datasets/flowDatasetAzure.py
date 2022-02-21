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

from __future__ import print_function

import numpy as np

from .DataLoaderBase import AzureDataLoader

# from .FilesystemHelper import get_filename_parts

class FlyingFlowDatasetAzure(AzureDataLoader):
    def load_image(self, fn):
        """Overload the base class. """
        img   = None
        count = 0

        while img is None and count < self.maxTrial:
            img = self.download_image(fn)
            count += 1

        if img is None:
            print('Image {} cannot load from Azure data lake storage. '.format(fn))

        return img

    def load_flow(self, fn, maskFn=None):
        """Overload the base class. """
        flow  = None
        count = 0

        while flow is None and count < self.maxTrial:
            p = self.download_pfm(fn)

            if ( p is not None ):
                flow = p[0][:, :, 0:2]

            count += 1

        if flow is None:
            print('Image {} cannot load from Azure data lake storage. '.format(fn))

        if ( maskFn is not None ):
            print("FlyingFlowDatasetAzure: mask file {} is not valid. This dataset contains no masks. ".format( \
                maskFn ))

        return flow, None

    def load_motion(self, idx):
        return None

class SintelFlowDatasetAzure(AzureDataLoader):
    def load_image(self, fn):
        """Overload the base class. """
        img   = None
        count = 0

        while img is None and count < self.maxTrial:
            img = self.download_image(fn)
            count += 1

        if img is None:
            print('Image {} cannot load from Azure data lake storage. '.format(fn))

        return img

    def load_flow(self, fn, maskFn=None):
        """Overload the base class. """
        flow  = None
        count = 0

        while flow is None and count < self.maxTrial:
            flow = self.download_flo(fn)
            count += 1

        if flow is None:
            print('Image {} cannot load from Azure data lake storage. '.format(fn))

        if ( maskFn is not None ):
            print("SintelFlowDatasetAzure: mask file {} is not valid. This dataset contains no masks. ".format( \
                maskFn ))

        return flow, None

    def load_motion(self, idx):
        return None

class TartanFlowDatasetAzure(AzureDataLoader):
    def load_image(self, fn):
        """Overload the base class. """
        img   = None
        count = 0

        while img is None and count < self.maxTrial:
            img = self.download_image(fn)
            count += 1

        if img is None:
            print('Image {} cannot load from Azure data lake storage. '.format(fn))

        return img

    def make_flow_mask_filename(self, fn):
        """fn is the flow file name. """

        # Get the filename components of fn.
        # parts  = get_filename_parts(fn)
        # maskFn = parts[1]
        # maskFn = maskFn.replace('_flow', '_mask')

        return fn.replace('flow.npy', 'mask.npy') # "%s/%s.npy" % ( parts[0], maskFn )

    def load_flow(self, fn, maskFn=None):
        """Overload the base class. """
        flow  = None
        count = 0

        while flow is None and count < self.maxTrial:
            flow = self.download_npy(fn)
            count += 1

        if flow is None:
            print('Image {} cannot load from Azure data lake storage. '.format(fn))

        if ( maskFn is not None ):
            # print(maskFn)
            mask = self.download_npy(maskFn)
            mask = mask.astype(np.uint8)
            mask_cross = np.zeros_like(mask).astype(np.uint8) # 0 - valid, 255 - cross occlusion
            mask_cross[mask == 1] = 255
        else:
            mask_cross = None

        return flow.astype(np.float32), mask_cross

    def load_motion(self, idx):
        if ( self.motions is not None ):
            return self.motions[idx]
        else:
            return None

if __name__ == '__main__':
    print('flowDatasetAzure.py! ')

    import argparse
    import torch
    from FLO import read_flo
    from utils import visflow
    import cv2
    from data_roots import *

    parser = argparse.ArgumentParser(description="Local test the data loader. ")

    parser.add_argument('infilelist', type=str, \
        help="The file list file. ")

    parser.add_argument('--pose-file', type=str, default='', \
        help="The filename of the pose file. ")

    # sampleDir = 'Sample_Sintel'

    args = parser.parse_args()

    dataset_term = args.infilelist.split('/')[-1].split('.txt')[0].split('_')[0]
    platform = 'azure'
    dataroot_list = FLOW_DR[dataset_term][platform]

    if ( args.pose_file != '' ):
        motionFn = args.pose_file
    else:
        motionFn = None

    dataloader = TartanFlowDatasetAzure( \
        args.infilelist, \
        rgb_root=dataroot_list[0], flow_root=dataroot_list[1], \
        has_mask=True, \
        motionFn=motionFn, \
        transform=None )
    
    print("len(dataloader) = {}".format( len(dataloader) ))

    for k in range(1):
        dataDict = dataloader[k]

        ffvis=visflow(dataDict['flow'])
        img1 = dataDict['img1']
        mask = dataDict['fmask']
        if ( motionFn is not None ):
            print('motion = {}'.format(dataDict['motion']))
        print('mask.shape = {}, mask.sum() = {}'.format(mask.shape, mask.sum()))
        zeros = mask == 0
        print('zeros.sum() = {}'.format(zeros.sum()))
        cv2.imshow('img', np.concatenate((ffvis,img1)))
        cv2.waitKey(0)

    # # Load a flow file from the local filesystem.
    # trueFlow = read_flo('%s/frame_0001.flo' % (sampleDir))
    # trueFlow = np.ascontiguousarray(trueFlow.transpose(2, 0, 1))
    # trueFlow = torch.from_numpy(trueFlow)

    # diff = trueFlow - dataDict['flow']

    # print('np.linalg.norm(diff) = {}'.format(np.linalg.norm(diff)))
    # print( 'torch.norm(diff) = {}'.format( torch.norm(diff) ) )

    # dataloaderNumPy = SintelFlowDatasetAzure( \
    #     args.infilelist, \
    #     args.acc_str_env, args.container, \
    #     transform=None )

    # dataDictNumPy = dataloaderNumPy[0]

    # img0 = cv2.imread('%s/frame_0001.png' % (sampleDir), cv2.IMREAD_UNCHANGED)
    # diffImg = img0 - dataDictNumPy['img1']

    # print('np.linalg.norm(diffImg) = {}'.format( np.linalg.norm(diffImg) ))

    # cv2.imwrite("%s/im0.png" % (sampleDir), dataDictNumPy['img1'])
    # cv2.imwrite("%s/im1.png" % (sampleDir), dataDictNumPy['img2'])
