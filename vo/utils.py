import torch

# def loadPretrain(model, preTrainModel):
#     preTrainDict = torch.load(preTrainModel)
#     model_dict = model.state_dict()
#     print 'preTrainDict:',preTrainDict.keys()
#     print 'modelDict:',model_dict.keys()
#     preTrainDict = {k:v for k,v in preTrainDict.items() if k in model_dict}
#     for item in preTrainDict:
#         print '  Load pretrained layer: ',item
#     model_dict.update(preTrainDict)
#     # for item in model_dict:
#     #   print '  Model layer: ',item
#     model.load_state_dict(model_dict)
#     return model

# from __future__ import division
# import torch
import math
import random
# from PIL import Image, ImageOps
import numpy as np
import numbers
import cv2

# ===== general functions =====

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class RandomCrop(object):
    """Crops the given imgage(in numpy format) at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    The image should be in shape: (h, w) or (h, w, c)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        kks = list(sample)
        th, tw = self.size
        for kk in kks:
            if len(sample[kk].shape)==3:
                h, w = sample[kk].shape[0], sample[kk].shape[1]
                break
        th = min(th, h)
        tw = min(tw, w)
        if w == tw and h == th:
            return sample
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape)==3:
                sample[kk] = img[y1:y1+th,x1:x1+tw,:]
            elif len(img.shape)==2:
                sample[kk] = img[y1:y1+th,x1:x1+tw]

        return sample

class CropCenter(object):
    """Crops the a sample of data (tuple) at center
    The image size should be larger than the target size, otherwise, the target size could not be reached
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        kks = list(sample)
        th, tw = self.size
        h, w = sample[kks[0]].shape[0], sample[kks[0]].shape[1]
        th = min(th, h)
        tw = min(tw, w)
        if w == tw and h == th:
            return sample

        x1 = int((w-tw)/2)
        y1 = int((h-th)/2)

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape)==3:
                sample[kk] = img[y1:y1+th,x1:x1+tw,:]
            elif len(img.shape)==2:
                sample[kk] = img[y1:y1+th,x1:x1+tw]

        return sample

class CropCenterNP(object):
    """Crops the given image(in numpy format) at center
    The image size should be larger than the target size, 
    otherwise, the target size could not be reached
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        th, tw = self.size
        h, w = sample.shape[0], sample.shape[1]
        th = min(th, h)
        tw = min(tw, w)
        if w == tw and h == th:
            return sample

        x1 = int((w-tw)/2)
        y1 = int((h-th)/2)

        if len(sample.shape)==3:
            sample = sample[y1:y1+th,x1:x1+tw,:]
        elif len(sample.shape)==2:
            sample = sample[y1:y1+th,x1:x1+tw]

        # # for debugging
        # resizeh = 480
        # resizew = 640
        # sample = cv2.resize(sample, (resizew,resizeh), interpolation=cv2.INTER_LINEAR)
        # sample[:,:,0] = sample[:,:,0] * (float(resizew)/float(tw))
        # sample[:,:,1] = sample[:,:,1] * (float(resizeh)/float(th))

        return sample

class ResizeFlowNP(object):
    """Resize the np array and scale the value
    """

    def __init__(self, size, scale_flow=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_flow = scale_flow

    def __call__(self, sample):

        th, tw = self.size
        h, w = sample.shape[0], sample.shape[1]
        sample = cv2.resize(sample, (tw,th), interpolation=cv2.INTER_LINEAR)
        if self.scale_flow:
            sample[:,:,0] = sample[:,:,0] * (float(tw)/float(w))
            sample[:,:,1] = sample[:,:,1] * (float(th)/float(h))

        return sample


class Combine12(object):
    '''
    combine the left and right images for stereo
    combine the first and second images for flow
    '''
    def __call__(self, sample):
        img1, img2 = sample['img1'], sample['img2']
        rbgs = torch.cat((img1, img2),dim=0)

        res = { 'rgbs':  rbgs}
        kks = list(sample)

        for kk in kks:
            if kk == 'img1' or kk == 'img2':
                continue
            res[kk] = sample[kk]

        return res

class ToTensor(object):
    def __call__(self, sample):
        kks = list(sample)

        for kk in kks:
            data = sample[kk] 
            if len(data.shape) == 3: # transpose image-like data
                data = data.transpose(2,0,1)
            elif len(data.shape) == 2:
                data = data.reshape((1,)+data.shape)
            sample[kk] = torch.from_numpy(data.copy()).float() # copy to make memory continuous

        return sample


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    this should be called after ToTensor
    """

    def __init__(self, mean, std, rgbbgr=False):
        self.mean = mean
        self.std = std
        self.rgbbgr = rgbbgr

    def __call__(self, sample):

        kks = list(sample)
        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape) == 3 and img.shape[0]==3: 
                img = img / float(255.0)
                if self.rgbbgr:
                    img = img[[2,1,0],:,:] # bgr2rgb
                for t, m, s in zip(img, self.mean, self.std): 
                    t.sub_(m).div_(s)
                sample[kk] = img
        return sample

class RandomHSV(object):
    """
    Change the image in HSV space
    """

    def __init__(self, HSVscale=(6,30,30), random_random=0):
        '''
        random_random > 0: different images use different HSV x% of the original random HSV value
        '''
        self.Hscale, self.Sscale, self.Vscale = HSVscale
        self.random_random = random_random


    def __call__(self, sample):
        kks = list(sample)
        # change HSV
        h = (random.random()*2-1) * self.Hscale
        s = (random.random()*2-1) * self.Sscale
        v = (random.random()*2-1) * self.Vscale

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape) == 3 and img.shape[2]==3: 
                imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                if self.random_random > 0: # add more noise to h s v
                    hh = (random.gauss(0, 0.5) * self.random_random + 1 ) * h
                    ss = (random.gauss(0, 0.5) * self.random_random + 1 ) * s
                    vv = (random.gauss(0, 0.5) * self.random_random + 1 ) * v
                else:
                    hh, ss, vv = h, s, v 
                imghsv[:,:,0] = np.clip(imghsv[:,:,0]+hh,0,255)
                imghsv[:,:,1] = np.clip(imghsv[:,:,1]+ss,0,255)
                imghsv[:,:,2] = np.clip(imghsv[:,:,2]+vv,0,255)

                sample[kk] = cv2.cvtColor(imghsv,cv2.COLOR_HSV2BGR)
        return sample

class FlipFlow(object):
    """
    Change the image in HSV space
    """

    def __init__(self):
        '''
        random_random > 0: different images use different HSV x% of the original random HSV value
        '''
        pass


    def __call__(self, sample):
        kks = list(sample)
        
        flipud, fliplr = False, False
        if random.random()>0.5:
            flipud = True
        if random.random()>0.5:
            fliplr = True

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape) == 3 and img.shape[2]==3: 
                if flipud:
                    img = np.flipud(img)
                if fliplr:
                    img = np.fliplr(img)

                sample[kk] = img

        if  'flow' in sample:
            if flipud:
                sample['flow'] = np.flipud(sample['flow'])
                sample['flow'][:,:,1] = -sample['flow'][:,:,1]
            if fliplr:
                sample['flow'] = np.fliplr(sample['flow'])
                sample['flow'][:,:,0] = -sample['flow'][:,:,0]

        if  'fmask' in sample:
            if flipud:
                sample['fmask'] = np.flipud(sample['fmask'])
            if fliplr:
                sample['fmask'] = np.fliplr(sample['fmask'])

        return sample

# ===== transform for stereo =====

class CombineLR(object):
    '''
    combine the left and right images for stereo
    '''
    def __call__(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        rbgs = torch.cat((leftImg, rightImg),dim=0)
        return { 'rgbs':  rbgs, 'disp': sample['disp']}


# ===== transform for flowvo ======

class RandomIntrinsic(object):
    """Crops the given image(in numpy format) at a random location to have a region of
    random size. Then resize the image to the original size. \
    Keep center: crop around the center pixel
    """

    def __init__(self, minlen=100, keep_center=False, fix_ratio=False):
        self.minlen = minlen
        self.keep_center = keep_center
        self.fix_ratio = fix_ratio

    def __call__(self, flownp):
        (h, w, c) = flownp.shape
        th = random.randint(self.minlen, h) # target h
        if self.fix_ratio:
            tw = int(th / float(h) * float(w)) 
        else:
            tw = random.randint(self.minlen, w) # target w

        if self.keep_center:
            x1 = int((w-tw)/2)
            y1 = int((h-th)/2)
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
        # print (x1, y1), (x1+tw, y1+th), (tw, th), (float(w)/float(tw)), (float(h)/float(th))

        flowcrop = flownp[y1:y1+th,x1:x1+tw,:]
        flowcrop = cv2.resize(flowcrop, (w,h), interpolation=cv2.INTER_LINEAR)

        # scale the flow
        flowcrop[:,:,0] = flowcrop[:,:,0] * (float(w)/float(tw))
        flowcrop[:,:,1] = flowcrop[:,:,1] * (float(h)/float(th))
        # print flowcrop.shape

        return flowcrop

def generate_random_scale_crop(h, w, target_h, target_w, scale_base, keep_center, fix_ratio):
    '''
    Randomly generate scale and crop params
    H: input image h
    w: input image w
    target_h: output image h
    target_w: output image w
    scale_base: max scale up rate
    keep_center: crop at center
    fix_ratio: scale_h == scale_w
    '''
    scale_w = random.random() * (scale_base - 1) + 1
    if fix_ratio:
        scale_h = scale_w
    else:
        scale_h = random.random() * (scale_base - 1) + 1

    crop_w = int(math.ceil(target_w/scale_w)) # ceil for redundancy
    crop_h = int(math.ceil(target_h/scale_h)) # crop_w * scale_w > w

    if keep_center:
        x1 = int((w-crop_w)/2)
        y1 = int((h-crop_h)/2)
    else:
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)

    return scale_w, scale_h, x1, y1, crop_w, crop_h

class RandomIntrinsic2(object):
    """
    Random scale to cover contineous focal length
    Due to the tartanair focal is already small, we only up scale the image

    """

    def __init__(self, size, max_focal=800.0, keep_center=False, fix_ratio=False):
        if isinstance(size, numbers.Number):
            self.th, self.tw = (int(size), int(size))
        else:
            self.th, self.tw = size

        self.max_focal = max_focal
        self.keep_center = keep_center
        self.fix_ratio = fix_ratio
        self.tartan_focal = 320.

        assert self.max_focal >= self.tartan_focal
        self.scale_base = self.max_focal /self.tartan_focal

    def __call__(self, flownp):
        (h, w, c) = flownp.shape

        # target_focal_w = random.random() * (self.max_focal - self.tartan_focal) + self.tartan_focal
        # scale_w = target_focal_w/self.tartan_focal
        # if self.fix_ratio:
        #     scale_h = scale_w
        # else:
        #     target_focal_h = random.random() * (self.max_focal - self.tartan_focal) + self.tartan_focal
        #     scale_h = target_focal_h/self.tartan_focal


        # crop_w = int(math.ceil(w/scale_w)) # ceil for redundancy
        # crop_h = int(math.ceil(h/scale_h)) # crop_w * scale_w > w

        # if crop_w >= w and crop_h >= h:
        #     return flownp

        # if self.keep_center:
        #     x1 = int((w-crop_w)/2)
        #     y1 = int((h-crop_h)/2)
        # else:
        #     x1 = random.randint(0, w - crop_w)
        #     y1 = random.randint(0, h - crop_h)

        scale_w, scale_h, x1, y1, crop_w, crop_h = generate_random_scale_crop(h, w, self.th, self.tw, 
                                                    self.scale_base, self.keep_center, self.fix_ratio)

        flowcrop = flownp[y1:y1+crop_h,x1:x1+crop_w,:]
        flowcrop = cv2.resize(flowcrop, (0,0), fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
        assert flowcrop.shape[0] >= self.th
        assert flowcrop.shape[1] >= self.tw
        # print scale, (x1, y1), (x1+crop_w, y1+crop_h), flowcrop.shape

        # scale the flow
        flowcrop[:,:,0] = flowcrop[:,:,0] * scale_w
        flowcrop[:,:,1] = flowcrop[:,:,1] * scale_h
        # print flowcrop.shape

        return flowcrop[:self.th,:self.tw,:]

class RandomIntrinsicBatch(object):
    """
    Random scale to cover contineous focal length
    Due to the tartanair focal is already small, we only up scale the image

    """

    def __init__(self, batchsize, size, max_focal=800.0, keep_center=False, fix_ratio=False):
        self.batchsize = batchsize
        self.randomIntri = RandomIntrinsic2(size, max_focal=max_focal, keep_center=keep_center, fix_ratio=fix_ratio)

    def __call__(self, flownp):
        reslist = []
        for k in range(self.batchsize):
            sample = self.randomIntri(flownp)
            reslist.append(sample)
        
        return np.array(reslist)


class KittiTransform(object):
    """
    This transform is mainly for debugging
    1. resize the height to target height and target scale w
    2. crop center to target size
    """

    def __init__(self, size, scale_w):
        if isinstance(size, numbers.Number):
            self.target_h, self.target_w = (int(size), int(size))
        else:
            self.target_h, self.target_w = size
        self.cropCenter = CropCenterNP(size)
        self.scale_w = scale_w

    def __call__(self, flownp):
        (h, w, c) = flownp.shape
        scale_h = float(self.target_h)/h
        flowresize = cv2.resize(flownp, (0,0), fx=self.scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
        flowresize[:,:,0] = flowresize[:,:,0] * self.scale_w
        flowresize[:,:,1] = flowresize[:,:,1] * scale_h

        return self.cropCenter(flowresize)

class Tartan2Kitti2(object):
    def __init__(self):
        focal_t = 320.0
        cx_t = 320.0
        cy_t = 240
        focal_k = 707.0912
        cx_k = 601.8873
        cy_k = 183.1104


        self.th = 370
        self.tw = 1226

        self.sx = (0-cx_k)/focal_k*focal_t+cx_t
        self.sy = (0-cy_k)/focal_k*focal_t+cy_t

        self.step = focal_t/ focal_k
        self.scale = focal_k/ focal_t

    def __call__(self, flownp):
        # import ipdb;ipdb.set_trace()
        res = np.zeros((self.th, self.tw, flownp.shape[-1]), dtype=np.float32)
        thh = 0.
        for hh in range(self.th):
            tww = 0.
            for ww in range(self.tw):
                res[hh, ww, :] = bilinear_interpolate(flownp, self.sy+thh, self.sx+tww)
                tww += self.step
            thh += self.step

        res = res * self.scale

        return res

class Tartan2Kitti(object):
    def __init__(self):
        pass

    def __call__(self, flownp):
        # import ipdb;ipdb.set_trace()
        flowcrop = flownp[157:325 ,48:603, :]
        flowcrop = cv2.resize(flowcrop, (1226,370), interpolation=cv2.INTER_LINEAR)
        # scale the flow
        flowcrop[:,:,0] = flowcrop[:,:,0] * (2.209)
        flowcrop[:,:,1] = flowcrop[:,:,1] * (2.202)

        return flowcrop


# ========= end-to-end flow and vo ==========
class RandomResizeCrop(object):
    """
    Random scale to cover continuous focal length
    Due to the tartanair focal is already small, we only up scale the image

    """

    def __init__(self, size, max_focal=800.0, keep_center=False, fix_ratio=False):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        if isinstance(size, numbers.Number):
            self.target_h = int(size)
            self.target_w = int(size)
        else:
            self.target_h = size[0]
            self.target_w = size[1]

        self.max_focal = max_focal
        self.keep_center = keep_center
        self.fix_ratio = fix_ratio
        self.tartan_focal = 320.

        assert self.max_focal >= self.tartan_focal
        self.scale_base = self.max_focal /self.tartan_focal

    def __call__(self, sample): 
        flownp =  sample['flow']
        h, w, _ = flownp.shape

        scale_w, scale_h, x1, y1, crop_w, crop_h = generate_random_scale_crop(h, w, self.target_h, self.target_w, 
                                                    self.scale_base, self.keep_center, self.fix_ratio)

        for kk in ['flow', 'img1', 'img2', 'intrinsic', 'fmask']:
            if kk in sample:
                sample[kk] = sample[kk][y1:y1+crop_h, x1:x1+crop_w]
                sample[kk] = cv2.resize(sample[kk], (0,0), fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
                # Note opencv reduces the last dimention if it is one
                sample[kk] = sample[kk][:self.target_h,:self.target_w]

        # scale the flow
        if  'flow' in sample:
            sample['flow'][:,:,0] = sample['flow'][:,:,0] * scale_w
            sample['flow'][:,:,1] = sample['flow'][:,:,1] * scale_h

        return sample


class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size

    """
    def __init__(self, scale=4):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        self.downscale_flow = scale

    def __call__(self, sample): 
        flownp =  sample['flow']
        h, w, _ = flownp.shape

        # scale the flow
        sample['flow'] = cv2.resize(sample['flow'], 
            (int(w/self.downscale_flow), int(h/self.downscale_flow)), interpolation=cv2.INTER_LINEAR)

        if self.downscale_flow!=1 and  'intrinsic' in sample:
            sample['intrinsic'] = cv2.resize(sample['intrinsic'], 
                (int(w/self.downscale_flow), int(h/self.downscale_flow)), interpolation=cv2.INTER_LINEAR)

        if self.downscale_flow!=1 and  'fmask' in sample:
            sample['fmask'] = cv2.resize(sample['fmask'],
                (int(w/self.downscale_flow), int(h/self.downscale_flow)), interpolation=cv2.INTER_LINEAR)
            
        return sample

def tensor2img(tensImg,mean,std):
    """
    convert a tensor a numpy array, for visualization
    """
    # undo normalize
    for t, m, s in zip(tensImg, mean, std):
        t.mul_(s).add_(m) 
    tensImg = tensImg * float(255)
    # undo transpose
    tensImg = (tensImg.numpy().transpose(1,2,0)).astype(np.uint8)
    return tensImg

def bilinear_interpolate(img, h, w):
    # assert round(h)>=0 and round(h)<img.shape[0]
    # assert round(w)>=0 and round(w)<img.shape[1]

    h0 = int(math.floor(h))
    h1 = h0 + 1
    w0 = int(math.floor(w))
    w1 = w0 + 1

    a = h - h0 
    b = w - w0

    h0 = max(h0, 0)
    w0 = max(w0, 0)
    h1 = min(h1, img.shape[0]-1)
    w1 = min(w1, img.shape[1]-1)

    A = img[h0,w0,:]
    B = img[h1,w0,:]
    C = img[h0,w1,:]
    D = img[h1,w1,:]

    res = (1-a)*(1-b)*A + a*(1-b)*B + (1-a)*b*C + a*b*D

    return res 

def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return bgr


# ========= ADJUST CAMERA INTRINSICS =======


def resize_image_and_intrinsics(img, intrinsics, newSize, scale_value=False):
    """newSize is in (H, W) order. """
    
    H, W = img.shape[0:2]

    img = cv2.resize( img, ( newSize[1], newSize[0] ), interpolation=cv2.INTER_LINEAR )

    sH = 1.0 * newSize[0] / H
    sW = 1.0 * newSize[1] / W

    intrinsics = intrinsics.copy()

    intrinsics[0, 0] *= sW
    intrinsics[1, 1] *= sH

    intrinsics[0, 2] *= sW
    intrinsics[1, 2] *= sH

    if scale_value: # Ture when the image comes from optical flow
        assert len(img.shape) == 3
        assert img.shape[2] >= 2
        img[:,:,0] = img[:,:,0] * sW
        img[:,:,1] = img[:,:,1] * sH

    return img, intrinsics

class ROI(object):
    def __init__(self, x0, y0, dx, dy):
        super(ROI, self).__init__()

        self.x0 = x0
        self.y0 = y0
        self.x1 = x0 + dx
        self.y1 = y0 + dy
        self.dx = dx
        self.dy = dy

def find_roi_around_principle_point(W, H, cx, cy):
    hX = int( min( np.floor(cx), np.floor(W - cx - 1) ) )
    hY = int( min( np.floor(cy), np.floor(H - cy - 1) ) )

    return ROI( cx - hX, cy - hY, 2*hX, 2*hY )

def max_sample_around_principle_point(img, intrinsics):
    H, W = img.shape[0:2]

    # Find the ROI round the principle point.
    roi = find_roi_around_principle_point( 
        W, H, intrinsics[0, 2], intrinsics[1, 2] )

    rangeX = np.arange(0, roi.dx)
    rangeY = np.arange(0, roi.dy)

    mapX, mapY = np.meshgrid( rangeX, rangeY )
    mapX = mapX.astype(np.float32)
    mapY = mapY.astype(np.float32)
    mapX = mapX + roi.x0
    mapY = mapY + roi.y0

    dst = cv2.remap( img, mapX, mapY, cv2.INTER_LINEAR )

    intrinsics = intrinsics.copy()
    intrinsics[0, 2] = roi.dx / 2
    intrinsics[1, 2] = roi.dy / 2

    return dst, intrinsics

def resized_maxed_sample_around_principle_point(img, intrinsics, newH):
    imgMax, newK = max_sample_around_principle_point(img, intrinsics)

    r = newK[0, 0] / newK[1, 1]

    newW = newH / imgMax.shape[0] * imgMax.shape[1] / r

    imgScaled, kScaled = resize_image_and_intrinsics( imgMax, newK, ( int(newH), int(newW) ) )

    return imgScaled, kScaled

class AdjustIntrinsics(object):
    """
    Resize and crop the images to make sure: Fx == Fy, Ox = w/2, Oy = h/2

    """

    def __init__(self, intrinsics, newH):
        self.intrinsics = intrinsics
        self.newH = newH

    def adjust_flow(self, flow):
        imgMax, newK = max_sample_around_principle_point(flow, self.intrinsics)

        r = float(newK[0, 0]) / newK[1, 1]

        newW = float(self.newH) / imgMax.shape[0] * imgMax.shape[1] / r

        imgScaled, kScaled = resize_image_and_intrinsics( imgMax, newK, ( int(self.newH), int(newW) ) , scale_value=True)

        return imgScaled

    def adjust_tuple(self, sample):
        pass

    def __call__(self, sample): 
        if type(sample) is tuple:
            return self.adjust_tuple(sample)
        else:
            return self.adjust_flow(sample)

def dataset_intrinsics(dataset='tartanair'):
    if dataset == 'kitti':
        focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
    elif dataset == 'euroc':
        focalx, focaly, centerx, centery = 355.6358642578, 417.1617736816, 362.2718811035, 249.6590118408
    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0
    else:
        return None
    return focalx, focaly, centerx, centery

def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)
    return intrinsicLayer

if __name__ == '__main__':
    testflow = np.random.rand(50,30,4).astype(np.float32) * 5
    randintrinsic = RandomIntrinsic(minlen=10)
    cropflow = randintrinsic.__call__(testflow)
