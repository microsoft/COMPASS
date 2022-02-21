from __future__ import print_function

import numpy as np
import os
import io
import cv2

from torch.utils.data import Dataset
from .utils import make_intrinsics_layer

from .pfm import readPFM_Bytes
from .FLO import read_flo_bytes

class DatasetBase(Dataset):
    '''
    Loader for multi-modal data
    -----
    framelistfile: 
    TRAJNAME FRAMENUM
    FRAMESTR0
    FRAMESTR1
    ...
    -----
    Requirements: 
    The actural data path consists three parts: DATAROOT+TRAJNAME+CONVERT(FRAMESTR)
    The frames under the same TRAJNAME should be consequtive. So when the next frame is requested, it chould return the next one in the same sequence. 
    The last frame of the trajectory should not in the list, but it is existed on the harddrive, in case the next frame is requested. 
    '''
    def __init__(self, \
        framelistfile, \
        datatypes = "img0,img1,disp0,disp1,flow,flow2,motion,img0n,img1n,disp0n,disp1n", \
        imgdir="", flowdir="", depthdir="", \
        motionFn=None, norm_trans=False, \
        transform=None, \
        flow_norm = 1., has_mask=False, flowinv = False, \
        disp_norm = 1., \
        pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013], \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0):

        super(DatasetBase, self).__init__()

        self.trajlist, self.trajlenlist, self.framelist = self.parse_inputfile(framelistfile)
        self.datatypelist = datatypes.split(',')

        self.framelistfile = framelistfile
        self.imgroot = imgdir 
        self.flowroot = flowdir
        self.depthroot = depthdir

        self.flow_norm = flow_norm
        self.disp_norm = disp_norm
        self.pose_norm = pose_norm
        self.transform = transform

        self.flagFlowMask = has_mask
        self.flowinv = flowinv
        self.intrinsic = intrinsic
        self.focalx    = focalx
        self.focaly    = focaly
        self.centerx   = centerx
        self.centery   = centery

        self.N = len(self.framelist)
        self.trajnum = len(self.trajlenlist)
        self.acc_trajlen = np.cumsum(self.trajlenlist)

    def parse_inputfile(self, inputfile):
        '''
        trajlist: [TRAJ0, TRAJ1, ...]
        trajlenlist: [TRAJLEN0, TRAJLEN1, ...]
        framelist: [FRAMESTR0, FRAMESTR1, ...]
        '''
        with open(inputfile,'r') as f:
            lines = f.readlines()
        trajlist, trajlenlist, framelist = [], [], []
        ind = 0
        while ind<len(lines):
            line = lines[ind].strip()
            traj, trajlen = line.split(' ')
            trajlen = int(trajlen)
            trajlist.append(traj)
            trajlenlist.append(trajlen)
            ind += 1
            for k in range(trajlen):
                if ind>=len(lines):
                    print("Datafile Error: {}, line {}...".format(self.framelistfile, ind))
                    raise Exception("Datafile Error: {}, line {}...".format(self.framelistfile, ind))
                line = lines[ind].strip()
                framelist.append(line)
                ind += 1

        print('Read {} trajectories, including {} frames'.format(len(trajlist), len(framelist)))
        return trajlist, trajlenlist, framelist

    def getDataPath(self, trajstr, framestr, datatype):
        raise NotImplementedError

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        raise NotImplementedError

    def load_image(self, fn):
        raise NotImplementedError

    def load_disparity(self, fn):
        raise NotImplementedError

    def load_motion(self, idx):
        raise NotImplementedError

    def idx2traj(self, idx):
        for k in range(self.trajnum):
            if idx < self.acc_trajlen[k]:
                break
        return self.trajlist[k]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # parse the idx to trajstr
        trajstr = self.idx2traj(idx)
        framestr = self.framelist[idx]

        sample = {}
        h, w = None, None
        for datatype in self.datatypelist:
            datafile = self.getDataPath(trajstr, framestr, datatype)
            if datatype == 'img0' or datatype == 'img1' or datatype == 'img0n' or datatype == 'img1n':
                img = self.load_image(datafile)
                if img is None:
                    print("!!!READ IMG ERROR {}, {}, {}".format(idx, trajstr, framestr, datafile))
                sample[datatype] = img
                h, w, _ = img.shape
            if datatype == 'disp0' or datatype == 'disp1' or datatype == 'disp0n' or datatype == 'disp1n':
                disp = self.load_disparity(datafile)
                disp = disp * self.disp_norm
                sample[datatype] = disp
                h, w = disp.shape
            if datatype == 'flow' or datatype == 'flow2':
                flow, mask = self.load_flow(datafile)
                flow = flow * self.flow_norm
                sample[datatype] = flow
                if self.flagFlowMask:
                    masktype = datatype.replace('flow','fmask')
                    sample[masktype] = mask
                h, w, _ = flow.shape
            if datatype == 'motion':
                motion = self.load_motion(idx)
                sample[datatype] = motion

        if self.intrinsic:
            if h is None or w is None:
                Exception("Unknow Input H/W {}".format(self.datatypelist))
            intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
            sample['intrinsic'] = intrinsicLayer

        # Transform.
        if ( self.transform is not None):
            sample = self.transform(sample)

        return sample

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

class AzureDataLoaderBase():
    def __init__(self, ):
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
                cv2.IMREAD_UNCHANGED )
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