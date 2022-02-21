import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os.path import isfile, join, isdir
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses

class TartanTrajFlowDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, flowfolder = 'testfolder', posefile = None, transform = None, flow_norm = 0.05, norm_trans=False, 
                    intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0):
        
        files = listdir(flowfolder)
        self.flowfiles = [(flowfolder +'/'+ ff) for ff in files if ff.endswith('flow.npy')]
        self.flowfiles.sort()

        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32)

        print('Find {} flow files in {}'.format(len(self.flowfiles), flowfolder))

        # import ipdb;ipdb.set_trace()
        if posefile is not None:
            poselist = np.loadtxt(posefile).astype(np.float32)
            poses = pos_quats2SEs(poselist)
            self.matrix = pose2motion(poses)
            self.motions     = SEs2ses(self.matrix).astype(np.float32)
            self.motions = self.motions / self.pose_std
            assert(len(self.motions) == len(self.flowfiles))
        else:
            self.motions = None

        self.N = len(self.flowfiles)

        self.flow_norm = flow_norm

        # self.N = len(self.lines)
        self.transform = transform
        self.intrinsic = intrinsic
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        flowname = self.flowfiles[idx].strip()
        flow = np.load(flowname).astype(np.float32)
        flow = flow * self.flow_norm

        # import ipdb;ipdb.set_trace()
        if self.intrinsic:
            h, w, _ = flow.shape
            ww, hh = np.meshgrid(range(w), range(h))
            ww = (ww.astype(np.float32) - self.centerx + 0.5 )/self.focalx
            hh = (hh.astype(np.float32) - self.centery + 0.5 )/self.focaly
            intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)
            flow = np.concatenate((flow, intrinsicLayer), axis=2)            

        if self.transform:
            flow = self.transform(flow)
        flow = flow.transpose(2,0,1)

        if self.motions is None:
            return flow
        else:
            return {'flow': flow, 'motion': self.motions[idx], 'matrix': self.matrix[idx] }

class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder , posefile = None, 
                    transform = None, norm_trans=False, 
                    intrinsic = False, focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
        
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if ff.endswith('.png')]
        self.rgbfiles.sort()
        self.imgfolder = imgfolder

        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32)

        print('Find {} image files in {}'.format(len(self.rgbfiles), imgfolder))

        if posefile is not None:
            poselist = np.loadtxt(posefile).astype(np.float32)
            poses = pos_quats2SEs(poselist)
            self.matrix = pose2motion(poses)
            self.motions     = SEs2ses(self.matrix).astype(np.float32)
            self.motions = self.motions / self.pose_std
            assert(len(self.motions) == len(self.rgbfiles)) - 1
        else:
            self.motions = None

        self.N = len(self.rgbfiles) - 1

        # self.N = len(self.lines)
        self.transform = transform
        self.intrinsic = intrinsic
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        imgfile1 = self.rgbfiles[idx].strip()
        imgfile2 = self.rgbfiles[idx+1].strip()
        img1 = cv2.imread(join(self.imgfolder, imgfile1))
        img2 = cv2.imread(join(self.imgfolder, imgfile2))

        res = {'img1': img1, 'img2': img2 }
        # import ipdb;ipdb.set_trace()
        if self.intrinsic:
            h, w, _ = img1.shape
            ww, hh = np.meshgrid(range(w), range(h))
            ww = (ww.astype(np.float32) - self.centerx + 0.5 )/self.focalx
            hh = (hh.astype(np.float32) - self.centery + 0.5 )/self.focaly
            intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)
            res['intrinsic'] = intrinsicLayer

        if self.transform:
            res = self.transform(res)

        if self.motions is None:
            return res
        else:
            res['motion'] = self.motions[idx]
            return res


if __name__ == '__main__':

    # import time
    from utils import KittiTransform
    test_traj_dir = '/home/wenshan/tmp/data/E03'
    transform = KittiTransform((448, 640), scale_w=0.5)

    # read intrinsics from file
    calibfile = test_traj_dir + '/calib.txt'
    with open(calibfile, 'r') as f:
        lines = f.readlines()
    cam_intrinsics = lines[2].strip().split(' ')[1:]
    focalx, focaly, centerx, centery = float(cam_intrinsics[0]), float(cam_intrinsics[5]), float(cam_intrinsics[2]), float(cam_intrinsics[6])
    print('Read camera intrinsics: {}, {}, {}, {}'.format(focalx, focaly, centerx, centery))

    trainDataset = TartanTrajFlowDataset(test_traj_dir+'/flow', None, transform=transform, 
                                        intrinsic=True, focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    for k in range(0, len(trainDataset),1):
        sample = trainDataset[k]
        from utils import visflow
        flowvis = visflow(sample.transpose(1,2,0)/trainDataset.flow_norm)
        cv2.imshow('img',flowvis)
        print(str(k))
        cv2.waitKey(0)
        # import ipdb;ipdb.set_trace()

    # from utils import KittiTransform
    # transform = KittiTransform()
    # trainDataset = TartanTrajFlowDataset('/home/wenshan/tmp/data/EuRoC/Undistorted/V1_03_difficult_mav0_Undistorted/cam0/flow', 
    #                                      # '/home/wenshan/tmp/data/kitti/vo/10/pose_left.txt',
    #                                      transform = transform,
    #                                      intrinsic=False, focalx=707.0912, focaly=707.0912, centerx=601.8873, centery=183.1104)
    # for k in range(133, len(trainDataset),2):
    #     sample = trainDataset[k]
    #     from utils import visflow
    #     flowvis = visflow(sample.transpose(1,2,0)*10)
    #     cv2.imshow('img',flowvis)
    #     print(str(k))
    #     cv2.waitKey(0)


    # from utils import CropCenter
    # eurocpath = '/home/wenshan/tmp/data/EuRoC/Undistorted/MH_01_easy_mav0_Undistorted/cam0/data2'
    # posefile = 'MH01/MH01_pose.txt'
    # eurocDataset = TrajFolderDataset(imgfolder = eurocpath, posefile = posefile,  transform = CropCenter((384,512)),
    #                                 intrinsic=True, focalx=355.6358642578, focaly=417.1617736816, 
    #                                 centerx=362.2718811035, centery=249.6590118408)

    # print(len(eurocDataset))

    # for k in range(1000):
    #     sample = eurocDataset[k]
    #     img1 = sample['img1']
    #     img2 = sample['img2']
    #     print sample['motion']
    #     cv2.imshow('img', np.concatenate((img1,img2),1))
    #     # import ipdb;ipdb.set_trace()
    #     cv2.waitKey(0)