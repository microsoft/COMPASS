import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(here+'/..') # add the parent path to the sys path

import torch 
import torch.nn as nn
import torch.nn.functional as F

class PretrainPoseNet(nn.Module):
    def __init__(self, pretrain='', intrinsic=False, down_scale=True, use_gru=False):
        super(PretrainPoseNet, self).__init__()

        from VOFlowNet import VOFlowRes as FlowPoseNet
        if use_gru:
            fcnum = 512
        else:
            fcnum = 256
        self.flowPoseNet = FlowPoseNet( down_scale=down_scale, fcnum = fcnum)
        self.use_gru = use_gru
         
        from backbone.select_backbone import select_resnet
        from backbone.convrnn import ConvGRU
        _, _, _, self.preEncoder, _ = select_resnet('resnet18', norm='bn') # , norm='none'
        
        if use_gru: # Add GRU Part
            self.agg_f = ConvGRU(input_size=256,
                               hidden_size=256,
                               kernel_size=1,
                               num_layers=1)
            self.agg_b = ConvGRU(input_size=256,
                               hidden_size=256,
                               kernel_size=1,
                               num_layers=1)

        if pretrain=='':
            print('Training from scratch')
        else:
            print('Using the pretrained encoder')
            self.init_weights(pretrain,use_gru)
            self.fix_normalization() # Note: comment this out if norm='none'




    def init_weights(self, pretrain, use_gru):

        ckpt = torch.load('./models/'+pretrain)['state_dict']
        ckpt2 = {}
        if use_gru:
            ckpt_gru_f = {}
            ckpt_gru_b = {}
        ckpt3 = {}
        for key in ckpt:
            if key.startswith('backbone_flow'):
                newkey = key.replace('backbone_flow.', '')
                ckpt2[newkey] = ckpt[key]

                # # adapt to new model for IROS
                # if newkey.find('n1') != -1:
                #     ckpt2[newkey.replace('n1.', 'bn1.')] = ckpt[key]
                # elif newkey.find('n2') != -1:
                #     ckpt2[newkey.replace('n2.', 'bn2.')] = ckpt[key]
                # else:
                #     ckpt2[key.replace('backbone_flow.', '')] = ckpt[key]
                # # --- end of adapt ---

            if use_gru:
                if key.startswith('agg_f_flow'):
                    ckpt_gru_f[key.replace('agg_f_flow.', '')] = ckpt[key]
                    # print key.replace('agg_f_flow.', '')

                if key.startswith('agg_b_flow'):
                    ckpt_gru_b[key.replace('agg_b_flow.', '')] = ckpt[key]
                    # print key.replace('agg_b_flow.', '')

        # import ipdb;ipdb.set_trace()
        if use_gru:
            self.agg_f.load_state_dict(ckpt_gru_f)
            self.agg_b.load_state_dict(ckpt_gru_b)

        # import ipdb;ipdb.set_trace()
        self.preEncoder.load_state_dict(ckpt2)
        print('load pretrain success')
       
    def fix_normalization(self):
        self.preEncoder.train()
        for module in self.preEncoder.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()  
            if isinstance(module, torch.nn.modules.LayerNorm):
                module.eval()

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        if self.use_gru:
            return self.forward_gru(x)

        feat = self.preEncoder(x.unsqueeze(2)) 
        feat = feat.mean(dim=(2,3,4))
        pose = self.flowPoseNet(feat)
        
        return pose

    def forward_gru(self, x):

        # import ipdb;ipdb.set_trace()
        feat = self.preEncoder(x.unsqueeze(2)) 
        # feat = feat.mean(dim=(2,3,4))

        feature = F.relu(feat)
        feature = feature.squeeze(2).unsqueeze(1)
        B, N, C, H, W = feature.shape

        context_forward, _ = self.agg_f(feature) # TODO: GRU size doesn't match
        context_forward = context_forward[:,-1,:].unsqueeze(1)
        context_forward = F.avg_pool3d(context_forward, (1, H, W), stride=1).squeeze(-1).squeeze(-1)

        feature_back = torch.flip(feature, dims=(1,))
        context_back, _ = self.agg_b(feature_back)
        context_back = context_back[:,-1,:].unsqueeze(1)
        context_back = F.avg_pool3d(context_back, (1, H, W), stride=1).squeeze(-1).squeeze(-1)

        feat = torch.cat([context_forward, context_back], dim=-1) # B,N,C=2C


        pose = self.flowPoseNet(feat)
        
        return pose


class PretrainedVONet(nn.Module):
    '''
    Input: 2 RGB images
    Structure: PWC + PoseNet
    PoseNet: Pretrained Flow-feature-extractor + two-heads
    Return: Up-to-scale camera motion
    '''
    def __init__(self, intrinsic=True, flowNormFactor=1.0, down_scale=True, fixflow=True, pretrain=False, use_gru=False):
        super(PretrainedVONet, self).__init__()

        from PWC import PWCDCNet as FlowNet
        self.flowNet     = FlowNet()
        self.flowPoseNet = PretrainPoseNet(pretrain=pretrain, intrinsic=intrinsic, down_scale=down_scale, use_gru=use_gru)

        self.intrinsic = intrinsic
        self.flowNormFactor = flowNormFactor
        self.down_scale = down_scale

        if fixflow:
            for param in self.flowNet.parameters():
                param.requires_grad = False

    def forward(self, x, only_flow=False, only_pose=False, gt_flow=False):
        '''
        x[0]: rgb frame t-1
        x[1]: rgb frame t
        x[2]: intrinsics
        x[3]: flow t-1 -> t (optional)
        '''
        if not only_pose: # forward flownet
            flow_out = self.flowNet(x[0:2])
            if only_flow:
                return flow_out

            if self.down_scale:
                flow = flow_out[0]
            else:
                flow = F.interpolate(flow_out[0], scale_factor=4, mode='bilinear', align_corners=True)
        else:
            assert(gt_flow) # when only_pose==True, we should provide gt-flow as input
            assert(len(x)>3)
            flow_out = None

        if gt_flow:
            flow_input = x[3]
        else:
            flow_input = flow * self.flowNormFactor

        if self.intrinsic:
            flow_input = torch.cat( ( flow_input, x[2] ), dim=1 )
        
        pose = self.flowPoseNet(flow_input)

        return flow_out, pose

    def get_flow_loss(self, netoutput, target, criterion, mask=None, small_scale=False):
        '''
        small_scale: the target flow and mask are down scaled (when in forward_vo)
        '''
        # netoutput 1/4, 1/8, ..., 1/32 size flow
        if mask is not None:
            return self.flowNet.get_loss_w_mask(netoutput, target, criterion, mask, small_scale=small_scale)
        else:
            return self.flowNet.get_loss(netoutput, target, criterion, small_scale=small_scale)

if __name__ == '__main__':
    
    voflownet = PretrainedVONet(intrinsic=False, flowNormFactor=1.0, down_scale=True, fixflow=True, pretrain=True) # 
    # voflownet.cuda()
    voflownet.eval()
    print (voflownet)
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    # x, y = np.ogrid[:448, :640]
    # # print x, y, (x+y)
    # img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(512 + 384)
    # img = img.astype(np.float32)
    # print (img.dtype)
    # imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    # intrin = imgInput[:,:2,:448,:640].copy()

    # imgTensor = torch.from_numpy(imgInput)
    # intrinTensor = torch.from_numpy(intrin)
    # print (imgTensor.shape)
    # stime = time.time()
    # for k in range(100):
    #     flow, pose = voflownet((imgTensor.cuda(), imgTensor.cuda(), intrinTensor.cuda()))
    #     # print flow.data.shape, pose.data.shape
    #     # print pose.data.cpu().numpy()
    #     # print time.time()-stime
    # print (time.time()-stime)/100
