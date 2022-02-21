
from torch.utils.data import DataLoader
from .utils import RandomCrop, RandomResizeCrop, RandomHSV, ToTensor, Normalize, Compose, Combine12, FlipFlow, ResizeData, CropCenter
import numpy as np
from .data_roots import *

class MultiDatasetsBase(object):

    def __init__(self, datafiles, datatypes, databalence, 
                       args, batch, workernum, 
                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                       shuffle=True):

        self.datafiles = datafiles.split(',')
        self.datatypes = datatypes.split(',')
        databalence = databalence.split(',')
        databalence = [int(tdb) for tdb in databalence]
        assert len(self.datafiles) == len(self.datatypes)
        self.numDataset = len(self.datafiles)
        self.loss_mask = [False] * self.numDataset
        self.batch = batch
        self.workernum = workernum
        self.shuffle = shuffle

        self.dataloders = []
        self.dataiters = []
        self.lossmasks = []
        self.datalens = []

        self.init_datasets( args, mean, std)

        # calculate the percentage
        if len(databalence)>1:
            assert len(databalence) == len(self.datalens)
            self.datalens = np.array(self.datalens) * np.array(databalence)
        self.accDataLens = np.cumsum(self.datalens).astype(np.float)/np.sum(self.datalens)    

    def init_datasets(self, args, mean, std):
        raise NotImplementedError

    def load_sample(self):
        # Randomly pick the dataset in the list
        randnum = np.random.rand()
        datasetInd = 0 
        while randnum > self.accDataLens[datasetInd]: # 
            datasetInd += 1

        # load sample from the dataloader
        try:
            sample = self.dataiters[datasetInd].next()
            if sample[list(sample.keys())[0]].shape[0] < self.batch:
                self.dataiters[datasetInd] = iter(self.dataloders[datasetInd])
                sample = self.dataiters[datasetInd].next()
        except StopIteration:
            self.dataiters[datasetInd] = iter(self.dataloders[datasetInd])
            sample = self.dataiters[datasetInd].next()

        return sample, self.lossmasks[datasetInd]

class FlowMultiDatasets(MultiDatasetsBase):
    '''
    Load data from different sources
    '''
    def init_datasets(self, args, mean=None, std=None):

        if args.platform=='azure':
            from flowDatasetAzure import SintelFlowDatasetAzure as SintelFlowDataset
            from flowDatasetAzure import FlyingFlowDatasetAzure as FlyingFlowDataset
            from flowDatasetAzure import ChairsFlowDatasetAzure as ChairsFlowDataset
            from TartanAirDatasetAzure import TartanAirDatasetAzureNoCompress as TartanAirFlowDataset
        else:
            from flowDataset import SintelFlowDataset, FlyingFlowDataset, ChairsFlowDataset
            if args.compressed: # load compressed data
                from TartanAirDataset import TartanAirDataset as TartanAirFlowDataset
            else:
                from TartanAirDataset import TartanAirDatasetNoCompress as TartanAirFlowDataset

        for datafile, datatype in zip(self.datafiles, self.datatypes):
            image_height,image_width = args.image_height, args.image_width
            max_intrinsic = args.random_intrinsic
            lossmask = False
            flowtype = 'flow'
            if datatype == 'sintel':
                DataSetType = SintelFlowDataset
                image_height = min(image_height, 384) # hard code
            elif datatype == 'chair':
                DataSetType = ChairsFlowDataset
                image_height = min(image_height, 384) # hard code
                image_width = min(image_width, 512)
            elif datatype == 'flying' or datatype == 'flyinginv':
                image_height = max(image_height, 512) # hard code - decrease the difficulty of flyingthings, could cause out-of-mem error
                image_width = max(image_width, 640)
                max_intrinsic = min(370, max_intrinsic)
                DataSetType = FlyingFlowDataset
            # elif datatype == 'kitti': # TODO
            #     DataSetType = KITTIFlowDataset
                # lossmask = True
            elif datatype == 'tartan':
                DataSetType = TartanAirFlowDataset
                lossmask = True # set to False for now, the dynamic objects sometime are not shown in depth image
            elif datatype == 'tartan2':
                DataSetType = TartanAirFlowDataset
                lossmask = True # set to False for now, the dynamic objects sometime are not shown in depth image
                flowtype = 'flow2'
            else:
                print ('unknow train datatype {}!!'.format(datatype))
                assert False
            # import ipdb;ipdb.set_trace()
            dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
            platform = args.platform
            dataroot_list = FLOW_DR[dataset_term][platform]

            if max_intrinsic>0:
                transformlist = [ RandomResizeCrop(size=(image_height,image_width), max_scale=max_intrinsic/320.0, 
                                                    keep_center=args.random_crop_center, fix_ratio=args.fix_ratio) ]
            else:
                transformlist = [ RandomCrop(size=(image_height,image_width)) ] 

            if not args.no_data_augment:
                transformlist.append(RandomHSV((10,80,80), random_random=args.hsv_rand))
                transformlist.append(FlipFlow())

            transformlist.extend([Normalize(mean=mean,std=std),ToTensor()])
            if args.combine_lr:
                transformlist.append(Combine12())

            dataset = DataSetType(args.working_dir + '/' + datafile, 
                                    datatypes = "img0,img0n,"+flowtype, 
                                    imgdir=dataroot_list[0], flowdir=dataroot_list[1],
                                    transform=Compose(transformlist), 
                                    flow_norm = args.normalize_output, has_mask=lossmask, 
                                    flowinv = (datatype[-3:] == 'inv'))

            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            dataiter = iter(dataloader)
            datalen = len(dataset)
            self.dataloders.append(dataloader)
            self.dataiters.append(dataiter)
            self.lossmasks.append(lossmask)
            self.datalens.append(datalen)

            print("Load dataset {}, data type {}, data size {}, crop size ({}, {})".format(datafile, datatype,datalen,image_height,image_width))


class EndToEndMultiDatasets(MultiDatasetsBase):
    '''
    Load data from different sources
    '''
    def init_datasets(self, args, mean=None, std=None):

        # import ipdb;ipdb.set_trace()
        if args.platform=='azure':
            from .TartanAirDatasetAzure import TartanAirDatasetAzureNoCompress as DataSetType
        else:
            if args.compressed: # load compressed data
                from .TartanAirDataset import TartanAirDataset as DataSetType
            else:
                from .TartanAirDataset import TartanAirDatasetNoCompress as DataSetType

        for datafile, datatype in zip(self.datafiles, self.datatypes):
            image_height,image_width = args.image_height, args.image_width
            lossmask = False

            if datatype == 'tartan': # only tartan is supported for now
                lossmask = True 
                flowtype = 'flow'
            elif datatype == 'tartan2':
                lossmask = True
                flowtype = 'flow2'
            elif datatype == 'euroc':
                lossmask = False
                flowtype = 'flow'
                from e.urocDataset import EurocDataset as DataSetType
            elif datatype == 'kitti':
                lossmask = False
                flowtype = 'flow'
                from .kittiDataset import KittiVODataset as DataSetType
            else:
                print ('unknow train datatype {}!!'.format(datatype))
                assert False

            dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
            platform = args.platform
            dataroot_list = FLOW_DR[dataset_term][platform]

            if args.random_intrinsic>0:
                transformlist = [ RandomResizeCrop(size=(image_height,image_width), max_scale=args.random_intrinsic/320.0, 
                                                    keep_center=args.random_crop_center, fix_ratio=args.fix_ratio) ]
            else:
                if args.random_crop_center:
                    transformlist = [CropCenter(size=(image_height,image_width))]
                elif args.resize_no_crop:
                    transformlist = [ResizeData(size=(image_height, image_width))]
                else:
                    transformlist = [ RandomCrop(size=(image_height,image_width)) ]

            if args.downscale_flow:
                from Datasets.utils import DownscaleFlow
                transformlist.append(DownscaleFlow())

            if not args.no_data_augment:
                transformlist.append(RandomHSV((10,80,80), random_random=args.hsv_rand))
            transformlist.extend([Normalize(mean=mean,std=std),ToTensor()])

            if args.no_gt: # when testing trajectory, no gt file is available
                datastr = "img0,img0n"
                motionFn = None
            else:
                if args.vo_gt_flow == 1.0: # only load flow and motion
                    datastr = "motion," +flowtype
                else:
                    datastr = "img0,img0n,motion,"+flowtype
                motionFn = args.working_dir + '/' + datafile.replace('.txt','.npy')
                if datatype=='kitti':
                    motionFn = args.working_dir + '/' + datafile.replace('_flow.txt','_motion.npy') # hard code
                    if args.vo_gt_flow == 1.0: # only load flow and motion
                        datastr = "flow,motion" 
                    else:
                        datastr = "img0,img0n,motion,"+flowtype

            norm_trans = True
            dataset = DataSetType(args.working_dir + '/' + datafile, 
                                datatypes = datastr, \
                                imgdir = dataroot_list[0], flowdir = dataroot_list[1], 
                                motionFn = motionFn, norm_trans=norm_trans, 
                                transform=Compose(transformlist), 
                                flow_norm = args.normalize_output, has_mask=lossmask,
                                intrinsic=args.intrinsic_layer)

            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            dataiter = iter(dataloader)
            datalen = len(dataset)
            self.dataloders.append(dataloader)
            self.dataiters.append(dataiter)
            self.lossmasks.append(lossmask)
            self.datalens.append(datalen)

            print("Load dataset {}, data type {}, data size {}, crop size ({}, {})".format(datafile, datatype,datalen,image_height,image_width))

if __name__ == '__main__':

    class ARGS(object):
        def __init__(self):
            self.image_height = 480
            self.image_width = 640
            self.no_data_augment = False
            self.combine_lr = False
            self.platform = 'local'
            self.working_dir = '.'
            self.image_scale = 1 
            self.random_intrinsic = 800
            self.random_crop_center = False
            self.fix_ratio = False
            self.normalize_output = 0.05
            self.intrinsic_kitti = False
            self.norm_trans_loss = False
            self.linear_norm_trans_loss = True
            self.downscale_flow = True
            self.intrinsic_layer = False
            self.azure = False

    args = ARGS()

    # ===== Test FlowPoseMultiDatasets =====
    from utils import visflow, tensor2img
    import cv2
    trainDataloader = EndToEndMultiDatasets('data/tartan_flow_rgbs3,data/tartan_flow_rgbs3', 
                                            'tartan,tartan',  '1,2',
                                            args, 2, 1, mean=[0., 0., 0.],std=[1., 1., 1.])
    # import ipdb;ipdb.set_trace()
    for k in range(100):
        sample, has_mask = trainDataloader.load_sample()
        flownp = sample['flow'].numpy()
        if has_mask:
            masknp = sample['fmask'].numpy()

        flowvis = visflow(flownp[0].transpose(1,2,0) / args.normalize_output)
        img1 = tensor2img(sample['img1'][0],mean=[0., 0., 0.],std=[1., 1., 1.])
        img2 = tensor2img(sample['img2'][0],mean=[0., 0., 0.],std=[1., 1., 1.])

        flowvis_2 = visflow(flownp[1].transpose(1,2,0) / args.normalize_output)
        img1_2 = tensor2img(sample['img1'][1],mean=[0., 0., 0.],std=[1., 1., 1.])
        img2_2 = tensor2img(sample['img2'][1],mean=[0., 0., 0.],std=[1., 1., 1.])

        if has_mask:
            flowvis[masknp[0][0]>128] = 0
            flowvis_2[masknp[1][0]>128] = 0

        imgdisp1 = np.concatenate((flowvis, flowvis_2), axis=0)
        if args.downscale_flow:
            imgdisp1 = cv2.resize(imgdisp1, (0,0), fx=4, fy=4)
        imgdisp2 = np.concatenate((img1, img1_2), axis=0)
        imgdisp3 = np.concatenate((img2, img2_2), axis=0)
        # import ipdb;ipdb.set_trace()
        imgdisp = cv2.resize(np.concatenate((imgdisp1, imgdisp2, imgdisp3), axis=1),(0,0),fx=0.5,fy=0.5)
        cv2.imshow('img',imgdisp)
        cv2.waitKey(0)
