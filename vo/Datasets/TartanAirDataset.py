from __future__ import print_function

import cv2
import numpy as np

from .DatasetBase import DatasetBase
from .utils import flow16to32, depth_rgba_float32

class TartanAirDatasetBase(DatasetBase):
    def __init__(self, \
        framelistfile, \
        datatypes = "img0,img1,disp0,disp1,flow,flow2,motion,img0n,img1n,disp0n,disp1n", \
        imgdir="", flowdir="", depthdir="", \
        motionFn=None, norm_trans=False, \
        transform=None, \
        flow_norm = 1., has_mask=False, flowinv = False, \
        disp_norm = 1., \
        pose_norm = [ 0.13,0.13,0.13,0.013 ,0.013,0.013], \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0):

        super(TartanAirDatasetBase, self).__init__(framelistfile, \
                                                    datatypes, \
                                                    imgdir, flowdir, depthdir, \
                                                    motionFn, norm_trans, \
                                                    transform, \
                                                    flow_norm, has_mask, flowinv, \
                                                    disp_norm, \
                                                    pose_norm, \
                                                    intrinsic, focalx, focaly, centerx, centery)

        if motionFn is not None:
            self.motions = np.load(motionFn).astype(np.float32)
            assert len(self.motions)==self.N, 'Motion file length mismatch {}, other data: {}'.format(len(self.motions), self.N)
            self.motions = self.motions / self.pose_norm
            if norm_trans: # normalize the translation for monocular VO 
                trans = self.motions[:,:3]
                trans_norm = np.linalg.norm(trans, axis=1)
                self.motions[:,:3] = self.motions[:,:3]/trans_norm.reshape(-1,1)
        else:
            self.motions = None

    def getDataPath(self, trajstr, framestr, datatype):
        '''
        return the file path name wrt the data type and framestr
        '''
        return NotImplementedError

    def load_image(self, fn):
        img = cv2.imread(self.imgroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        assert img is not None, "Error loading image {}".format(fn)
        return img

    def load_motion(self, idx):
        return self.motions[idx]


class TartanAirDataset(TartanAirDatasetBase):
    def __init__(self, \
        framelistfile, \
        datatypes = "img0,img1,disp0,disp1,flow,flow2,motion,img0n,img1n,disp0n,disp1n", \
        imgdir="", flowdir="", depthdir="", \
        motionFn=None, norm_trans=False, \
        transform=None, \
        flow_norm = 1., has_mask=False, flowinv = False, \
        disp_norm = 1., \
        pose_norm = [ 0.13,0.13,0.13,0.013 ,0.013,0.013], \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0):

        super(TartanAirDataset, self).__init__(framelistfile, \
                                                datatypes, \
                                                imgdir, flowdir, depthdir, \
                                                motionFn, norm_trans, \
                                                transform, \
                                                flow_norm, has_mask, flowinv, \
                                                disp_norm, \
                                                pose_norm, \
                                                intrinsic, focalx, focaly, centerx, centery)

    def getDataPath(self, trajstr, framestr, datatype):
        '''
        return the file path name wrt the data type and framestr
        '''
        if datatype == 'img0':
            return trajstr + '/image_left/' + framestr + '_left.png'
        if datatype == 'img1':
            return trajstr + '/image_right/' + framestr + '_right.png'
        if datatype == 'disp0':
            return trajstr + '/depth_left/' + framestr + '_left_depth.png'
        if datatype == 'disp1':
            return trajstr + '/depth_right/' + framestr + '_right_depth.png'

        if self.flowinv:
            framestr2 = str(int(framestr) - 1).zfill(len(framestr))
            framestr3 = str(int(framestr) - 2).zfill(len(framestr))
        else:    
            framestr2 = str(int(framestr) + 1).zfill(len(framestr))
            framestr3 = str(int(framestr) + 2).zfill(len(framestr))

        if datatype == 'flow':
            return trajstr + '/flowpwc_acfalse/' + framestr + '_' + framestr2 + '_flow.png' # FOR TESTING
        if datatype == 'flow2':
            return trajstr + '/flow2/' + framestr + '_' + framestr3 + '_flow.png'

        if 'flow2' in self.datatypelist: # hard code - TODO: better way to handle flow flow2 flowinv
            if self.flowinv:
                framestr2 = str(int(framestr) - 2).zfill(len(framestr))
            else:
                framestr2 = str(int(framestr) + 2).zfill(len(framestr))
        if datatype == 'img0n':
            return trajstr + '/image_left/' + framestr2 + '_left.png'
        if datatype == 'img1n':
            return trajstr + '/image_right/' + framestr2 + '_right.png'
        if datatype == 'disp0n':
            return trajstr + '/depth_left/' + framestr2 + '_left_depth.png'
        if datatype == 'disp1n':
            return trajstr + '/depth_right/' + framestr2 + '_right_depth.png'
        return None

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        flow16 = cv2.imread(self.flowroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        assert flow16 is not None, "Error loading flow {}".format(fn)
        flow32, mask = flow16to32(flow16)
        return flow32, mask

    def load_disparity(self, fn, fw=80.0):
        depth_rgba = cv2.imread(self.depthroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        assert depth_rgba is not None, "Error loading depth {}".format(fn)
        depth = depth_rgba_float32(depth_rgba)
        disp = fw/depth
        return disp


class TartanAirDatasetNoCompress(TartanAirDatasetBase):
    def __init__(self, \
        framelistfile, \
        datatypes = "img0,img1,disp0,disp1,flow,flow2,motion,img0n,img1n,disp0n,disp1n", \
        imgdir="", flowdir="", depthdir="", \
        motionFn=None, norm_trans=False, \
        transform=None, \
        flow_norm = 1., has_mask=False, flowinv = False, \
        disp_norm = 1., \
        pose_norm = [ 0.13,0.13,0.13,0.013 ,0.013,0.013], \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0):

        super(TartanAirDatasetNoCompress, self).__init__(framelistfile, \
                                                    datatypes, \
                                                    imgdir, flowdir, depthdir, \
                                                    motionFn, norm_trans, \
                                                    transform, \
                                                    flow_norm, has_mask, flowinv, \
                                                    disp_norm, \
                                                    pose_norm, \
                                                    intrinsic, focalx, focaly, centerx, centery)

    def getDataPath(self, trajstr, framestr, datatype):
        '''
        return the file path name wrt the data type and framestr
        '''
        if datatype == 'img0':
            return trajstr + '/image_left/' + framestr + '_left.png'
        if datatype == 'img1':
            return trajstr + '/image_right/' + framestr + '_right.png'
        if datatype == 'disp0':
            return trajstr + '/depth_left/' + framestr + '_left_depth.npy'
        if datatype == 'disp1':
            return trajstr + '/depth_right/' + framestr + '_right_depth.npy'

        if self.flowinv:
            framestr2 = str(int(framestr) - 1).zfill(len(framestr))
            framestr3 = str(int(framestr) - 2).zfill(len(framestr))
        else:    
            framestr2 = str(int(framestr) + 1).zfill(len(framestr))
            framestr3 = str(int(framestr) + 2).zfill(len(framestr))
        if datatype == 'flow':
            return trajstr + '/flowpwc_acfalse/' + framestr + '_' + framestr2 + '_flow.npy'  # FOR TESTING
        if datatype == 'flow2':
            return trajstr + '/flow2/' + framestr + '_' + framestr3 + '_flow.npy'

        if 'flow2' in self.datatypelist: # hard code - TODO: better way to handle flow flow2 flowinv
            if self.flowinv:
                framestr2 = str(int(framestr) - 2).zfill(len(framestr))
            else:
                framestr2 = str(int(framestr) + 2).zfill(len(framestr))
        else:
            if self.flowinv:
                framestr2 = str(int(framestr) - 1).zfill(len(framestr))
            else:
                framestr2 = str(int(framestr) + 1).zfill(len(framestr))            
        if datatype == 'img0n':
            return trajstr + '/image_left/' + framestr2 + '_left.png'
        if datatype == 'img1n':
            return trajstr + '/image_right/' + framestr2 + '_right.png'
        if datatype == 'disp0n':
            return trajstr + '/depth_left/' + framestr2 + '_left_depth.npy'
        if datatype == 'disp1n':
            return trajstr + '/depth_right/' + framestr2 + '_right_depth.npy'
        return None

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        flow = np.load(self.flowroot + '/' + fn)
        try:
            mask = np.load(self.flowroot + '/' + fn.replace('_flow.npy', '_mask.npy'))
        except:
            mask = np.ones((flow.shape[0], flow.shape[1])).astype(np.uint8)
        return flow, mask

    def load_disparity(self, fn, fw=80.0):
        depth = np.load(self.depthroot + '/' + fn)
        assert depth is not None, "Error loading depth {}".format(fn)
        disp = fw/depth
        return disp

if __name__ == '__main__':
    import time
    rootdir = '/home/wenshan/tmp/data/tartan'
    typestr = "img0,disp0,flow,img0n,disp0n,motion"

    dataset = TartanAirDataset(framelistfile='data/tartan_train_local_compress.txt',
                                    datatypes = typestr, \
                                    imgdir=rootdir, flowdir=rootdir, depthdir=rootdir,
                                    motionFn='data/tartan_flow_pose_local.npy', norm_trans=False, \
                                    transform=None, \
                                    flow_norm = 1., has_mask=True, \
                                    disp_norm = 1., \
                                    pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013], \
                                    intrinsic=True, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0)

    starttime = time.time()
    from utils import visflow, visdepth
    for k, sample in enumerate(dataset):
        print ('{}, {}, {}'.format(k, sample['motion'], sample['flow'].shape))
        d0 = sample['img0']
        d1 = visdepth(sample['disp0'])
        d2 = visflow(sample['flow'])
        d3 = visdepth(sample['disp0n'])
        dd = np.concatenate((np.concatenate((d0, d1), axis=0), np.concatenate((d2, d3), axis=0)), axis=1)
        dd = cv2.resize(dd, (640, 480))
        cv2.imshow('img',dd)
        cv2.waitKey(0)
    print (time.time() - starttime)

    # dataset = TartanAirDatasetNoCompress(framelistfile='data/tartan_train_local.txt',
    #                                 datatypes = typestr, \
    #                                 imgdir=rootdir, flowdir=rootdir, depthdir=rootdir,
    #                                 motionFn='data/tartan_flow_pose_local.npy', norm_trans=False, \
    #                                 transform=None, \
    #                                 flow_norm = 1., has_mask=True, \
    #                                 pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013], \
    #                                 intrinsic=True, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0)
    # starttime = time.time()
    # for k, sample in enumerate(dataset):
    #     print ('{}, {}, {}'.format(k, sample['motion'], sample['flow'].shape))
    # print (time.time() - starttime)
    import ipdb;ipdb.set_trace()