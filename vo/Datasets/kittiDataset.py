import cv2
from .DatasetBase import DatasetBase
from .pfm import readPFM
import numpy as np
from .utils import depth_rgba_float32, flow16to32

class KittiStereoDataset(DatasetBase):
    def __init__(self, \
        framelistfile, \
        datatypes = "img0,img1,disp0", \
        imgdir="", depthdir="", \
        transform=None):
        super(KittiStereoDataset, self).__init__(framelistfile, \
                                                datatypes = datatypes, \
                                                imgdir=imgdir, depthdir=depthdir,
                                                transform=transform)

    def getDataPath(self, trajstr, framestr, datatype):
        if datatype == 'img0':
            # return trajstr + '/colored_0/' + framestr + '_10.png'
            return trajstr + '/image_left/' + framestr + '.png'
        if datatype == 'img1':
            # return trajstr + '/colored_1/' + framestr + '_10.png'
            return trajstr + '/image_right/' + framestr + '.png'
        if datatype == 'disp0':
            return trajstr + '/disp_occ/' + framestr + '_10.png'
        return None

    def load_image(self, fn):
        # import ipdb;ipdb.set_trace()
        img = cv2.imread(self.imgroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        return img

    def load_disparity(self, fn):
        dispImg = cv2.imread(self.depthroot + '/' + fn)
        return dispImg[:,:,0].astype(np.float32)

class KittiVODataset(DatasetBase):
    def __init__(self, \
        framelistfile, \
        datatypes = "img0,img0n,img1,img1n,motion", \
        imgdir="", flowdir="", depthdir="", \
        motionFn = None, norm_trans=False, \
        transform=None, \
        flow_norm = 0.05, has_mask=False, \
        disp_norm = 0.05, \
        intrinsic = False, \
        focalx=707.0912, focaly=707.0912, centerx=601.8873, centery=183.1104):

        calibfile = framelistfile.split('.txt')[0].split('_flow')[0] + '_calib.txt' # hard-code
        with open(calibfile, 'r') as f:
            lines = f.readlines()
        cam_intrinsics = lines[2].strip().split(' ')[1:]
        focalx, focaly, centerx, centery = float(cam_intrinsics[0]), float(cam_intrinsics[5]), float(cam_intrinsics[2]), float(cam_intrinsics[6])
        self.focalx = focalx

        super(KittiVODataset, self).__init__(framelistfile, \
                                                datatypes = datatypes, 
                                                imgdir=imgdir, flowdir=flowdir, depthdir=depthdir,
                                                motionFn = motionFn, norm_trans=norm_trans, 
                                                transform=transform,
                                                flow_norm=flow_norm, disp_norm=disp_norm,
                                                intrinsic=intrinsic, 
                                                focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
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
        if datatype == 'img0':
            return trajstr + '/image_left/' + framestr + '.png'
        if datatype == 'img1':
            return trajstr + '/image_right/' + framestr + '.png'
        if datatype == 'img0n':
            framestr2 = str(int(framestr) + 1).zfill(6)
            return trajstr + '/image_left/' + framestr2 + '.png'
        if datatype == 'img1n':
            framestr2 = str(int(framestr) + 1).zfill(6)
            return trajstr + '/image_right/' + framestr2 + '.png'
        if datatype == 'disp0':
            return trajstr + '/disp4_3/' + framestr + '.png'
        if datatype == 'flow':
            framestr2 = str(int(framestr) + 1).zfill(6)
            return trajstr + '/flow/' + framestr + '_' + framestr2 + '_flow.png'
        return None

    def load_image(self, fn):
        img = cv2.imread(self.imgroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        return img

    def load_motion(self, idx):
        return self.motions[idx]

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        if fn.endswith('npy'):
            flow = np.load(self.flowroot + '/' + fn)
        elif fn.endswith('png'):
            flow16 = cv2.imread(self.flowroot + '/' + fn, cv2.IMREAD_UNCHANGED)
            flow, _ = flow16to32(flow16) 
        return flow, None

    def load_disparity(self, fn, fw=380.34):
        fw = self.focalx * 0.54
        disp_rgba = cv2.imread(self.depthroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        assert disp_rgba is not None, "Error loading depth {}".format(fn)
        disp = depth_rgba_float32(disp_rgba)
        disp_convert = disp / fw * 80.0
        return disp_convert

if __name__ == '__main__':
    kittidataset = KittiVODataset('data/kitti/kitti_00_flow.txt', \
                                imgdir='/bigdata/tartanvo_data/kitti/vo', \
                                motionFn='data/kitti/kitti_00_motion.npy', \
                                )


    for k in range(0,100):
        sample = kittidataset[k]
        leftimg = sample['img0']
        rightimg = sample['img1']
        leftimg2 = sample['img0n']
        rightimg2 = sample['img1n']


        img1 = np.concatenate((leftimg, rightimg),axis=0)
        img2 = np.concatenate((leftimg2, rightimg2),axis=0)
        img = np.concatenate((img1,img2),axis=1)
        img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        # import ipdb;ipdb.set_trace()

