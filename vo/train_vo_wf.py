import cv2
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.optim as optim
from workflow import WorkFlow, TorchFlow
from arguments import get_args
import numpy as np
from Datasets.data_roots import *
from Datasets.MultiDatasets import EndToEndMultiDatasets, FlowMultiDatasets
import random

# from scipy.io import savemat
np.set_printoptions(precision=4, threshold=10000, suppress=True)

import time # for testing

from PretrainedVONet import PretrainedVONet

class TrainVONet(TorchFlow.TorchFlow):
    def __init__(self, workingDir, args, prefix = "", suffix = "", plotterType = 'Visdom'):
        super(TrainVONet, self).__init__(workingDir, prefix, suffix, disableStreamLogger = False, plotterType = plotterType)
        self.args = args    
        self.saveModelName = 'vonet'

        # import ipdb;ipdb.set_trace()
        self.vonet = PretrainedVONet(intrinsic=self.args.intrinsic_layer, 
                            flowNormFactor=1.0, down_scale=args.downscale_flow, 
                            fixflow=args.fix_flow, pretrain=args.pretrain_model_name,
                            use_gru=args.use_gru)

        # load flow
        if args.load_flow_model:
            modelname1 = self.args.working_dir + '/models/' + args.flow_model
            if args.flow_model.endswith('tar'): # load pwc net
                data = torch.load(modelname1)
                self.vonet.flowNet.load_state_dict(data)
                print('load pwc network...')
            else:
                self.load_model(self.vonet.flowNet, modelname1)

        mean = None
        std = None

        self.pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013] # hard code, use when save motionfile when testing

        # load pose
        if args.load_pose_model:
            modelname2 = self.args.working_dir + '/models/' + args.pose_model
            self.load_model(self.vonet.flowPoseNet, modelname2)

        # load the whole model
        if self.args.load_model:
            modelname = self.args.working_dir + '/models/' + self.args.model_name
            self.load_model(self.vonet, modelname)

        self.LrDecrease = [int(self.args.train_step/2), 
                            int(self.args.train_step*3/4), 
                            int(self.args.train_step*7/8)]
        self.lr = self.args.lr
        self.lr_flow = self.args.lr_flow

        if not self.args.test: 
            if self.args.train_vo: # dataloader for end2end flow vo
                # import ipdb;ipdb.set_trace()
                self.trainDataloader = EndToEndMultiDatasets(self.args.data_file, self.args.train_data_type, self.args.train_data_balence, 
                                                args, self.args.batch_size, self.args.worker_num,  
                                                mean=mean, std=std)
                self.pretrain_lr = self.args.pretrain_lr_scale
                self.voflowOptimizer = optim.Adam([{'params':self.vonet.flowPoseNet.flowPoseNet.parameters(), 'lr': self.lr}, 
                                                    {'params':self.vonet.flowPoseNet.preEncoder.parameters(), 'lr': self.lr*self.pretrain_lr}], lr = self.lr)

            if self.args.train_flow: # dataloader for flow 
                self.trainFlowDataloader = FlowMultiDatasets(self.args.flow_file, self.args.flow_data_type, self.args.flow_data_balence,
                                                        self.args, self.args.batch_size, self.args.worker_num,
                                                        mean = mean, std = std)
                self.flowOptimizer = optim.Adam(self.vonet.flowNet.parameters(),lr = self.lr_flow)

            self.testDataloader = EndToEndMultiDatasets(self.args.val_file, self.args.test_data_type, '1',
                                                        self.args, self.args.batch_size, self.args.worker_num, 
                                                        mean=mean, std=std)
        else: 
            self.testDataloader = EndToEndMultiDatasets(self.args.val_file, self.args.test_data_type, '1',
                                                        self.args, self.args.batch_size, self.args.worker_num, 
                                                        mean=mean, std=std, shuffle= (not args.test_traj))


        self.criterion = nn.L1Loss()

        if self.args.multi_gpu>1:
            self.vonet = nn.DataParallel(self.vonet)

        self.vonet.cuda()

    def initialize(self):
        super(TrainVONet, self).initialize()

        self.AV['loss'].avgWidth = 100
        self.add_accumulated_value('flow', 100)
        self.add_accumulated_value('pose', 100)
        self.add_accumulated_value('vo_flow', 100)

        self.add_accumulated_value('test', 1)
        self.add_accumulated_value('t_flow', 1)
        self.add_accumulated_value('t_pose', 1)

        self.add_accumulated_value('t_trans', 1)
        self.add_accumulated_value('t_rot', 1)
        self.add_accumulated_value('trans', 100)
        self.add_accumulated_value('rot', 100)
        self.append_plotter("loss", ['loss', 'test'], [True, False])
        self.append_plotter("loss_flow", ['flow', 'vo_flow', 't_flow'], [True, True, False])
        self.append_plotter("loss_pose", ['pose', 't_pose'], [True, False])
        self.append_plotter("trans_rot", ['trans', 'rot', 't_trans', 't_rot'], [True, True, False, False])

        logstr = ''
        for param in self.args.__dict__.keys(): # record useful params in logfile 
            logstr += param + ': '+ str(self.args.__dict__[param]) + ', '
        self.logger.info(logstr) 

        self.count = 0
        self.test_count = 0
        self.epoch = 0

        super(TrainVONet, self).post_initialize()

    def dumpfiles(self):
        self.save_model(self.vonet, self.saveModelName+'_'+str(self.count))
        self.write_accumulated_values()
        self.draw_accumulated_values()

    def forward_flow(self, sample, use_mask=False): 
        # if self.args.combine_lr: # Not compatible w/ PWC yet!
        #     rgbs = sample['rgbs']
        #     output = self.vonet.flowNet(rgbs.cuda(), True)
        # else:
        img1Tensor = sample['img0'].cuda()
        img2Tensor = sample['img0n'].cuda()
        output = self.vonet([img1Tensor,img2Tensor], only_flow=True)
        targetflow = sample['flow'].cuda()

        # import ipdb;ipdb.set_trace()
        if not use_mask:
            mask = None
        else:
            mask = sample['fmask'].cuda()
        if self.args.multi_gpu>1:
            loss = self.vonet.module.get_flow_loss(output, targetflow, self.criterion, mask=mask)
        else:
            loss = self.vonet.get_flow_loss(output, targetflow, self.criterion, mask=mask) #flow_loss(output, targetflow, use_mask, mask)
        return loss/self.args.normalize_output, output

    def forward_vo(self, sample, use_mask=False):
        use_gtflow = random.random()<self.args.vo_gt_flow # use gt flow as the input of the posenet

        # load the variables
        if use_gtflow and self.args.fix_flow: # flownet is not trained, neither forward nor backward
            img0, img1 = None, None
            compute_flowloss = False
        else: 
            img0   = sample['img0'].cuda()
            img1   = sample['img0n'].cuda()
            compute_flowloss = True

        if self.args.intrinsic_layer:
            intrinsic = sample['intrinsic'].cuda()
        else: 
            intrinsic = None

        flow, mask = None, None
        if 'flow' in sample:
            flow = sample['flow'].cuda()
            if use_mask:
                mask = sample['fmask'].cuda()
        elif 'flow2' in sample:
            flow = sample['flow2'].cuda()
            if use_mask:
                mask = sample['fmask2'].cuda()

        if use_gtflow: # the gt flow will be input to the posenet
            # import ipdb;ipdb.set_trace()
            flow_output, pose_output = self.vonet([img0, img1, intrinsic, flow], only_pose=self.args.fix_flow, gt_flow=True)
        else: # use GT flow as the input
            flow_output, pose_output = self.vonet([img0, img1, intrinsic])
        pose_output_np = pose_output.data.cpu().detach().numpy().squeeze()

        if self.args.no_gt: 
            return 0., 0., 0., 0., pose_output_np

        # calculate flow loss
        if flow is not None and compute_flowloss:
            if self.args.multi_gpu>1:
                flowloss = self.vonet.module.get_flow_loss(flow_output, flow, self.criterion, mask=mask, small_scale=self.args.downscale_flow) /self.args.normalize_output
            else:
                flowloss = self.vonet.get_flow_loss(flow_output, flow, self.criterion, mask=mask, small_scale=self.args.downscale_flow) /self.args.normalize_output #flow_loss(flow_output, flow, use_mask, mask, small_scale=self.args.downscale_flow )/self.args.normalize_output
        else:
            flowloss = torch.FloatTensor([0])

        # calculate vo loss
        motion = sample['motion'].cuda()
        lossPose, trans_loss, rot_loss = self.linear_norm_trans_loss(pose_output, motion)

        return flowloss, lossPose, trans_loss, rot_loss, pose_output_np

    def linear_norm_trans_loss(self, output, motion, mask=None):
        output_trans = output[:, :3]
        output_rot = output[:, 3:]

        trans_norm = torch.norm(output_trans, dim=1).view(-1, 1)
        output_norm = output_trans/trans_norm

        if mask is None:
            trans_loss = self.criterion(output_norm, motion[:, :3])
            rot_loss = self.criterion(output_rot, motion[:, 3:])
        else:
            trans_loss = self.criterion(output_norm[mask,:], motion[mask, :3])
            rot_loss = self.criterion(output_rot[mask,:], motion[mask, 3:])

        loss = (rot_loss + trans_loss)/2.0

        return loss, trans_loss.item() , rot_loss.item()

    def train(self):
        super(TrainVONet, self).train()

        self.count = self.count + 1
        self.vonet.train()

        starttime = time.time()

        # train flow
        if self.args.train_flow: # not a vo only training
            flowsample, flowmask = self.trainFlowDataloader.load_sample()
            self.flowOptimizer.zero_grad()
            flowloss, _ = self.forward_flow(flowsample, use_mask=flowmask)
            flowloss.backward()
            self.flowOptimizer.step()
            self.AV['flow'].push_back(flowloss.item(), self.count)

        flowtime = time.time() 

        if self.args.train_vo: # not a flow only training
            self.voflowOptimizer.zero_grad()
            sample, vo_flowmask = self.trainDataloader.load_sample()
            loadtime = time.time()
            flowloss, poseloss, trans_loss, rot_loss, _ = self.forward_vo(sample, use_mask=vo_flowmask)
            if self.args.fix_flow:
                loss = poseloss
            else:
                loss = flowloss * self.args.lambda_flow + poseloss  # 
            loss.backward()
            self.voflowOptimizer.step()

            # import ipdb;ipdb.set_trace()
            self.AV['loss'].push_back(loss.item(), self.count)
            self.AV['vo_flow'].push_back(flowloss.item(), self.count)
            self.AV['pose'].push_back(poseloss.item(), self.count)
            self.AV['trans'].push_back(trans_loss, self.count)
            self.AV['rot'].push_back(rot_loss, self.count)

        nntime = time.time()

        # update Learning Rate
        if self.args.lr_decay:
            if self.count in self.LrDecrease:
                self.lr = self.lr*0.4
                self.lr_flow = self.lr_flow*0.4
                if self.args.train_vo:
                    assert len(self.voflowOptimizer.param_groups)==2
                    self.voflowOptimizer.param_groups[0]['lr'] = self.lr
                    self.voflowOptimizer.param_groups[1]['lr'] = self.lr * self.pretrain_lr
                if self.args.train_flow:
                    for param_group in self.flowOptimizer.param_groups: # ed_optimizer is defined in derived class
                        param_group['lr'] = self.lr_flow

        if self.count % self.args.print_interval == 0:
            losslogstr = self.get_log_str()
            self.logger.info("%s #%d - %s lr: %.6f - time (%.2f, %.2f)"  % (self.args.exp_prefix[:-1], 
                self.count, losslogstr, self.lr, flowtime-starttime, nntime-flowtime))

        if self.count % self.args.plot_interval == 0: 
            self.plot_accumulated_values()

        if self.count % self.args.test_interval == 0:
            if not (self.count)%self.args.snapshot==0:
                self.test()

        if (self.count)%self.args.snapshot==0:
            self.dumpfiles()
            # for k in range(self.args.test_num):
            #     self.test(save_img=True, save_surfix='test_'+str(k))

    def test(self):
        super(TrainVONet, self).test()
        self.test_count += 1

        self.vonet.eval()
        sample, mask = self.testDataloader.load_sample()

        with torch.no_grad():
            flowloss, poseloss, trans_loss, rot_loss, motion = self.forward_vo(sample, use_mask=mask)

        motion_unnorm = motion * self.pose_norm
        finish = self.test_count>= self.testDataloader.datalens[0]

        if self.args.no_gt:
            if self.test_count % self.args.print_interval == 0:
                self.logger.info("  TEST %s #%d - output : %s"  % (self.args.exp_prefix[:-1], 
                    self.test_count, motion_unnorm))
            return 0, 0, 0, 0, 0, motion_unnorm, finish

        if self.args.fix_flow:
            loss = poseloss
        else:
            loss = flowloss * self.args.lambda_flow  + poseloss # 

        lossnum = loss.item()
        self.AV['test'].push_back(lossnum, self.count)
        self.AV['t_flow'].push_back(flowloss.item(), self.count)
        self.AV['t_pose'].push_back(poseloss.item(), self.count)
        self.AV['t_trans'].push_back(trans_loss, self.count)
        self.AV['t_rot'].push_back(rot_loss, self.count)

        self.logger.info("  TEST %s #%d - (loss, flow, pose, rot, trans) %.4f  %.4f  %.4f  %.4f  %.4f"  % (self.args.exp_prefix[:-1], 
            self.test_count, loss.item(), flowloss.item(), poseloss.item(), rot_loss, trans_loss))

        return lossnum, flowloss.item(), poseloss.item(), trans_loss, rot_loss, motion_unnorm, finish

    def finalize(self):
        super(TrainVONet, self).finalize()
        if self.count < self.args.train_step and not self.args.test and not self.args.test_traj:
            self.dumpfiles()

        if self.args.test and not self.args.no_gt:
            self.logger.info('The average loss values: (t-trans, t-rot, t-flow, t-pose)')
            self.logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (self.AV['loss'].last_avg(100), 
                self.AV['t_trans'].last_avg(100),
                self.AV['t_rot'].last_avg(100),
                self.AV['t_flow'].last_avg(100),
                self.AV['t_pose'].last_avg(100)))

        else:
            self.logger.info('The average loss values: (trans, rot, test, t_trans, t_rot)')
            self.logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (self.AV['loss'].last_avg(100), 
                self.AV['trans'].last_avg(100),
                self.AV['rot'].last_avg(100),
                self.AV['test'].last_avg(100),
                self.AV['t_trans'].last_avg(100),
                self.AV['t_rot'].last_avg(100)))


if __name__ == '__main__':
    args = get_args()

    if args.use_int_plotter:
        plottertype = 'Int'
    else:
        plottertype = 'Visdom'
    try:
        # Instantiate an object for MyWF.
        trainVOFlow = TrainVONet(args.working_dir, args, prefix = args.exp_prefix, plotterType = plottertype)
        trainVOFlow.initialize()

        if args.test:
            errorlist = []
            motionlist = []
            finish = False
            while not finish:
                error0, error1, error2, error3, error4, motion, finish = trainVOFlow.test()
                errorlist.append([error0, error1, error2, error3, error4])
                motionlist.append(motion)
                if ( trainVOFlow.test_count == args.test_num ):
                    break
            errorlist = np.array(errorlist)
            print("Test reaches the maximum test number (%d)." % (args.test_num))
            print("Loss statistics: loss/flow/pose/trans/rot: (%.4f \t %.4f \t %.4f \t %.4f \t %.4f)" % (errorlist[:,0].mean(),
                            errorlist[:,1].mean(), errorlist[:,2].mean(), errorlist[:,3].mean(), errorlist[:,4].mean()))

            if args.test_traj:
                # save motion file
                outputdir_prefix = args.test_output_dir+'/'+args.model_name.split('vo')[0]+args.val_file.split('/')[-1].split('.txt')[0] # trajtest/xx_xx_euroc_xx
                motionfilename = outputdir_prefix +'_output_motion.txt'
                motions = np.array(motionlist)
                np.savetxt(motionfilename, motions)
                # visualize the file 
                # import ipdb;ipdb.set_trace()
                from error_analysis import evaluate_trajectory
                from evaluator.transformation import motion_ses2pose_quats, pose_quats2motion_ses
                from Datasets.utils import per_frame_scale_alignment
                gtposefile = args.gt_pose_file
                gtposes = np.loadtxt(gtposefile)
                gtmotions = pose_quats2motion_ses(gtposes)
                estmotion_scale = per_frame_scale_alignment(gtmotions, motions)
                estposes = motion_ses2pose_quats(estmotion_scale)
                evaluate_trajectory(gtposes, estposes, trajtype=args.test_data_type, outfilename=outputdir_prefix, scale=False, medir_dir=args.test_output_dir)
        else: # Training
            while True:
                trainVOFlow.train()
                if (trainVOFlow.count >= args.train_step):
                    break

        trainVOFlow.finalize()

    except WorkFlow.SigIntException as sie:
        print( sie.describe() )
        print( "Quit after finalize." )
        trainVOFlow.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")


