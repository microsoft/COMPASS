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

import cv2
import io
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from .pfm import readPFM_Bytes
from .FLO import read_flo_bytes

from .utils import make_intrinsics_layer

# def make_valid_azure_prefix(prefix):    
#     if ( len(prefix) > 0 and prefix[-1] != "/"):
#         return ''.join((prefix, "/"))
    
#     return prefix

def separate_lines_2_image_and_flow(lines, expected, \
    imgCPrefix="", flowCPrefix="", delimiter=' '):
    
    image0 = []
    image1 = []
    # imgCPrefix  = make_valid_azure_prefix(imgCPrefix)

    if( 3 == expected ):
        flow        = []
        # flowCPrefix = make_valid_azure_prefix(flowCPrefix)
    elif ( 2 == expected ):
        flow = None
    else:
        raise Exception("expected can only be 2 or 3. {} supplied. ".format(expected))

    for line in lines:
        ss = line.split(delimiter)

        if ( expected != len(ss) ):
            raise Exception("Line {} cannot be split into {} parts. ".format(\
                line, expected ))
        
        image0.append( "%s/%s" % ( imgCPrefix,  ss[0].strip() ) )
        image1.append( "%s/%s" % ( imgCPrefix,  ss[1].strip() ) )

        if ( 3 == expected ):
            flow.append(   "%s/%s" % ( flowCPrefix, ss[2].strip() ) )

    return image0, image1, flow


class DefaultTransform(object):
    def __init__(self):
        super(DefaultTransform, self).__init__()

        self.toTensor = ToTensor()

    def __call__(self, dict):
        t0 = self.toTensor( dict['img1'] )
        t1 = self.toTensor( dict['img2'] )

        flow   = torch.from_numpy( dict['flow'].transpose(2, 0, 1) )

        if ( dict['motion'] is not None ):
            motion = torch.from_numpy( dict['motion'] )
        else:
            motion = None

        return { \
            'img1': t0, 'img2': t1, \
            'flow': flow, 'motion': motion \
         }

class DataLoaderBase(Dataset):
    def __init__(self, \
        imgFlowListFn, imgCPrefix="", flowCPrefix="", \
        motionFn=None, norm_trans=False, \
        transform=DefaultTransform(), flow_norm = 1., flagFlow=True, \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0):

        super(DataLoaderBase, self).__init__()
        print(imgFlowListFn, os.path.exists(imgFlowListFn))
        if ( not os.path.isfile(imgFlowListFn) ):
            raise Exception("{} does not exist. ".format(imgFlowListFn))

        with open(imgFlowListFn,'r') as f:
            lines = f.readlines()

        imgFlowListDelimiter=" "
        if ( flagFlow ):
            self.imgFnList0, self.imgFnList1, self.flowFnList = \
                separate_lines_2_image_and_flow(lines, 3, imgCPrefix, flowCPrefix, imgFlowListDelimiter)
        else:
            self.imgFnList0, self.imgFnList1, self.flowFnList = \
                separate_lines_2_image_and_flow(lines, 2, imgCPrefix, flowCPrefix, imgFlowListDelimiter)
        
        self.N = len(self.imgFnList0)
        print('{} entries.'.format(self.N))

        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32)
        # self.flow_norm = flow_norm

        self.transform = transform

        # self.flagFlowMask = has_mask

        if motionFn is not None:
            self.motions = np.load(motionFn).astype(np.float32)
            assert(len(self.motions)==self.N)
            self.motions = self.motions / self.pose_std
            if norm_trans: # normalize the translation for monocular VO 
                trans = self.motions[:,:3]
                trans_norm = np.linalg.norm(trans, axis=1)
                self.motions[:,:3] = self.motions[:,:3]/trans_norm.reshape(-1,1)
        else:
            self.motions = None

        self.intrinsic = intrinsic
        self.focalx    = focalx
        self.focaly    = focaly
        self.centerx   = centerx
        self.centery   = centery

    def __len__(self):
        return self.N

    def make_flow_mask_filename(self, fn):
        """fn is the flow file name. """
        raise NotImplementedError

    def load_flow(self, fn, maskFn=None):
        """This function should return 2 objects, flow and mask. """
        raise NotImplementedError

    def load_image(self, fn):
        raise NotImplementedError

    def load_motion(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        imgFn0 = self.imgFnList0[idx]
        imgFn1 = self.imgFnList1[idx]

        img0 = self.load_image(imgFn0)
        img1 = self.load_image(imgFn1)

        if (img0 is None or img1 is None):
            
            raise Exception("img0 or img1 is None. idx = {}, imgFn0 = {}, imgFn1 = {}. ".format( \
                idx, imgFn0, imgFn1 ))
        
        # Compose the dictionary.
        tempDict = { 'img1': img0, 'img2': img1 }

        if ( self.flowFnList is not None ):
            flowFn = self.flowFnList[idx]

            #if ( self.flagFlowMask ):
            #    maskFn = self.make_flow_mask_filename(flowFn)
            #else:
            #    maskFn = None

            #flow, flowMask = self.load_flow(flowFn, maskFn)

            #if flow is None: # TODO: handle the error?
            #    raise Exception("flow is None. idx = {}, flowFn = {}. ".format( \
            #        idx, flowFn ))

            #flow = flow * self.flow_norm

            #tempDict['flow'] = flow

            #if ( flowMask is not None ):
            #    tempDict['fmask'] = flowMask

        if self.intrinsic:
            h, w, _ = img0.shape
            intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
            tempDict['intrinsic'] = intrinsicLayer

        # Before transforming.
        motion = self.load_motion(idx)

        if motion is not None:
            tempDict['motion'] = motion

        # Transform.
        if ( self.transform is not None):
            dataDict = self.transform(tempDict)
        else:
            dataDict = tempDict

        return dataDict

def get_container_client(connectionStr, containerName):
    from azure.storage.blob import BlobServiceClient, ContainerClient

    serviceClient   = BlobServiceClient.from_connection_string(connectionStr)
    containerClient = serviceClient.get_container_client(containerName)

    return containerClient, serviceClient

def get_azure_container_client(envString, containerString):
    # Get the connection string from the environment variable.
    connectStr = os.getenv(envString)
    # print(connectStr)

    # print("Get the container client. ")
    cClient, _ = get_container_client( connectStr, containerString )

    return cClient

class AzureDataLoader(DataLoaderBase):
    def __init__(self, \
        # accStrEnv, containerStr, \
        imgFlowListFn, imgCPrefix="", flowCPrefix="", \
        motionFn=None, norm_trans=False, \
        transform=DefaultTransform(), flow_norm = 1., flagFlow=True, has_mask=False, \
        intrinsic=False, focalX=320.0, focalY=320.0, centerX=320.0, centerY=240.0 ):

        super(AzureDataLoader, self).__init__( \
            imgFlowListFn, imgCPrefix, flowCPrefix, \
            motionFn, norm_trans, \
            transform, flow_norm, flagFlow, has_mask, \
            intrinsic, focalX, focalY, centerX, centerY )
        
        accStrEnv = 'AZURE_STORAGE_CONNECTION_STRING'
        containerStr = 'tartanairdataset'
        self.azCClient = get_azure_container_client(accStrEnv, containerStr)

        self.maxTrial = 10

    def set_max_trial(self, n):
        assert (n > 0), "n must be positive. {} encountered. ".format(n)

        self.maxTrial = n

    def download_npy(self, fn):
        '''
        return a numpy array given the file path in the Azure container.
        '''
        try:
            b = self.azCClient.get_blob_client(blob=fn)
            d = b.download_blob()
            e = io.BytesIO(d.content_as_bytes())
            f = np.load(e)
        except Exception as ex:
            print('npy: Exception: {}'.format(ex))
            return None

        return f

    def download_image(self, fn):
        try:
            b = self.azCClient.get_blob_client(blob=fn)
            d = b.download_blob()
            e = io.BytesIO(d.content_as_bytes())
            decoded = cv2.imdecode( \
                np.asarray( bytearray( e.read() ), dtype=np.uint8 ), \
                cv2.IMREAD_COLOR )
        except Exception as ex:
            print('image: Exception: {}'.format(ex))
            return None

        return decoded

    def download_pfm(self, fn):
        try:
            b = self.azCClient.get_blob_client(blob=fn)
            d = b.download_blob()
            e = io.BytesIO(d.content_as_bytes())
            p = readPFM_Bytes(e)
        except Exception as ex:
            print('pfm: Exception: {}'.format(ex))
            return None

        return p

    def download_flo(self, fn):
        try:
            b = self.azCClient.get_blob_client(blob=fn)
            d = b.download_blob()
            e = io.BytesIO(d.content_as_bytes())
            p = read_flo_bytes(e)
        except Exception as ex:
            print('flo: Exception: {}'.format(ex))
            return None

        return p

if __name__ == '__main__':
    print("Hello, Base.py! ")
