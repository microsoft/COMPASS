from DatasetBase import AzureDataLoaderBase
from TartanAirDataset import TartanAirDataset, TartanAirDatasetNoCompress, flow16to32, depth_rgba_float32


class TartanAirDatasetAzure(TartanAirDataset, AzureDataLoaderBase):
    def __init__(self,
        framelistfile, \
        datatypes = "img0,img1,disp0,disp1,flow,flow2,motion,img0n,img1n,disp0n,disp1n", \
        imgdir="", flowdir="", depthdir="", \
        motionFn=None, norm_trans=False, \
        transform=None, \
        flow_norm = 1., has_mask=False, \
        disp_norm = 1., \
        pose_norm = [ 0.13,0.13,0.13,0.013 ,0.013,0.013], \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0):

        TartanAirDataset.__init__(self, \
                                    framelistfile, \
                                    datatypes, \
                                    imgdir, flowdir, depthdir, \
                                    motionFn, norm_trans, \
                                    transform, \
                                    flow_norm, has_mask, \
                                    disp_norm, \
                                    pose_norm, \
                                    intrinsic, focalx, focaly, centerx, centery)
        AzureDataLoaderBase.__init__(self)


    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        flow16  = None
        count = 0

        while flow16 is None and count < self.maxTrial:
            flow16 = self.download_image(self.flowroot + '/' + fn)
            count += 1

        if flow is None:
            print('Flow {} cannot load from Azure data lake storage. '.format(fn))

        flow32, mask = flow16to32(flow16)
        return flow32, mask


    def load_disparity(self, fn, fw=80.0):
        depth_rgba  = None
        count = 0

        while depth_rgba is None and count < self.maxTrial:
            depth_rgba = self.download_image(self.depthroot + '/' + fn)
            count += 1

        if depth_rgba is None:
            print('Depth {} cannot load from Azure data lake storage. '.format(fn))

        depth = depth_rgba_float32(depth_rgba)
        disp = fw/depth
        return disp

    def load_image(self, fn):
        """Overload the base class. """
        img   = None
        count = 0

        while img is None and count < self.maxTrial:
            img = self.download_image(self.imgroot + '/' + fn)
            count += 1

        if img is None:
            print('Image {} cannot load from Azure data lake storage. '.format(fn))

        return img


class TartanAirDatasetAzureNoCompress(TartanAirDatasetNoCompress, AzureDataLoaderBase):
    def __init__(self,
        framelistfile, \
        datatypes = "img0,img1,disp0,disp1,flow,flow2,motion,img0n,img1n,disp0n,disp1n", \
        imgdir="", flowdir="", depthdir="", \
        motionFn=None, norm_trans=False, \
        transform=None, \
        flow_norm = 1., has_mask=False, \
        disp_norm = 1., \
        pose_norm = [ 0.13,0.13,0.13,0.013 ,0.013,0.013], \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0):

        TartanAirDatasetNoCompress.__init__(self, \
                                    framelistfile, \
                                    datatypes, \
                                    imgdir, flowdir, depthdir, \
                                    motionFn, norm_trans, \
                                    transform, \
                                    flow_norm, has_mask, \
                                    disp_norm, \
                                    pose_norm, \
                                    intrinsic, focalx, focaly, centerx, centery)
        AzureDataLoaderBase.__init__(self,)

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        flow  = None
        count = 0

        while flow is None and count < self.maxTrial:
            flow = self.download_npy(self.flowroot + '/' + fn)
            count += 1

        if flow is None:
            print('Flow {} cannot load from Azure data lake storage. '.format(fn))

        mask = None
        count = 0 
        maskFn = self.flowroot + '/' + fn.replace('_flow.npy', '_mask.npy')
        while mask is None and count < self.maxTrial:
            mask = self.download_npy(maskFn)
            count += 1

        return flow, mask


    def load_disparity(self, fn, fw=80.0):
        depth  = None
        count = 0

        while depth is None and count < self.maxTrial:
            depth = self.download_npy(self.depthroot + '/' + fn)
            count += 1

        if depth is None:
            print('Depth {} cannot load from Azure data lake storage. '.format(fn))

        disp = fw/depth
        return disp

    def load_image(self, fn):
        """Overload the base class. """
        img   = None
        count = 0

        while img is None and count < self.maxTrial:
            img = self.download_image(self.imgroot + '/' + fn)
            count += 1

        if img is None:
            print('Image {} cannot load from Azure data lake storage. '.format(fn))

        return img

if __name__ == '__main__':
    import time
    rootdir = ''
    typestr = "img0,disp0,flow,img0n,disp0n,motion"

    dataset = TartanAirDatasetAzureNoCompress(framelistfile='data/tartan_train.txt',
                                    datatypes = typestr, \
                                    imgdir=rootdir, flowdir=rootdir, depthdir=rootdir,
                                    motionFn='data/tartan_train_pose.npy', norm_trans=False, \
                                    transform=None, \
                                    flow_norm = 1., has_mask=True, \
                                    disp_norm = 1., \
                                    pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013], \
                                    intrinsic=True, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0)

    starttime = time.time()
    from utils import visflow, visdepth
    import numpy as np
    import cv2
    for k, sample in enumerate(dataset):
        print ('{}, {}, {}'.format(k, sample['motion'], sample['flow'].shape))
        if k%5==0:
            d0 = sample['img0']
            d1 = visdepth(sample['disp0'])
            d2 = visflow(sample['flow'])
            d3 = visdepth(sample['disp0n'])
            dd = np.concatenate((np.concatenate((d0, d1), axis=0), np.concatenate((d2, d3), axis=0)), axis=1)
            dd = cv2.resize(dd, (640, 480))
            cv2.imwrite(str(k)+'img.jpg',dd)
            # cv2.waitKey(0)
        if k==100:
            break
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