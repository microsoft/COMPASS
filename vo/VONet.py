
import torch 
import torch.nn as nn
import torch.nn.functional as F

class VONet(nn.Module):
    def __init__(self, args, network=0,  flowNormFactor=1.0, down_scale=True, config=1, fixflow=True):
        super(VONet, self).__init__()

        from VOFlowNet import VOFlowRes as FlowPoseNet
        self.flowPoseNet = FlowPoseNet( down_scale=down_scale)
         
        from backbone.select_backbone import select_resnet
        from backbone.convrnn import ConvGRU
        self.preEncoder, _, _, _, _ = select_resnet('resnet18') 
        
        if args.pre_train:
            print('Using the pretrained encoder')
            self.init_weights()
        else:
            print('Training from scratch')

    def init_weights(self):

        ckpt = torch.load('../../preMM_model/memdpc_zy_rotate_multimodal_zip-128_resnet18_mem1024_bs16_lr0.001_seq4_pred1_len1_ds1_segTrue_depthTrue/epoch600.pth.tar')['state_dict']
        ckpt2 = {}
        for key in ckpt:
            if key.startswith('backbone_rgb'):
                ckpt2[key.replace('backbone_rgb.', '')] = ckpt[key]

        self.preEncoder.load_state_dict(ckpt2)
        print('load pretrain success')
       
    def forward(self, x, only_flow=False):

        feat1, feat2 = self.preEncoder(x[0].unsqueeze(2)), self.preEncoder(x[1].unsqueeze(2))
        feat1, feat2 = feat1.mean(dim=(2,3,4)), feat2.mean(dim=(2,3,4))
        feat = torch.cat([feat1, feat2], dim=1)
        pose = self.flowPoseNet(feat)
        
        return pose

    def get_flow_loss(self, netoutput, target, criterion, mask=None, small_scale=False):
        if self.network == 0: # pwc net
            # netoutput 1/4, 1/8, ..., 1/32 size flow
            if mask is not None:
                return self.flowNet.get_loss_w_mask(netoutput, target, criterion, mask, small_scale=small_scale)
            else:
                return self.flowNet.get_loss(netoutput, target, criterion, small_scale=small_scale)
        else: 
            if mask is not None:
                valid_mask = mask<128
                valid_mask = valid_mask.expand(target.shape)
                return criterion(netoutput[valid_mask], target[valid_mask])
            else:
                return criterion(netoutput, target)

if __name__ == '__main__':
    
    voflownet = VONet(network=0, intrinsic=True, flowNormFactor=1.0, down_scale=True, config=1, fixflow=True) # 
    voflownet.cuda()
    voflownet.eval()
    print(voflownet)
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    x, y = np.ogrid[:448, :640]
    # print x, y, (x+y)
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(512 + 384)
    img = img.astype(np.float32)
    print(img.dtype)
    imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    intrin = imgInput[:,:2,:112,:160].copy()

    imgTensor = torch.from_numpy(imgInput)
    intrinTensor = torch.from_numpy(intrin)
    print(imgTensor.shape)
    stime = time.time()
    for k in range(100):
        flow, pose = voflownet((imgTensor.cuda(), imgTensor.cuda(), intrinTensor.cuda()))
        print(flow.data.shape, pose.data.shape)
        print(pose.data.cpu().numpy())
        print(time.time()-stime)
    print((time.time()-stime)/100)

