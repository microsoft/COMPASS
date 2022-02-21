import numpy as np
from evaluator.plot_trajectory import plot_gt_est_trajectories
from evaluator.tartanair_evaluator import TartanAirEvaluator
from evaluator.transformation import motion_ses2pose_quats, pose_quats2motion_ses, quats2SEs, SE2quat
from evaluator.trajectory_transform import shift0
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import join

from Datasets.utils import per_frame_scale_alignment


def plot_traj(gtposes, estposes, vis=False, savefigname=None, figsize=(4,4)):
    fig = plt.figure(figsize=figsize)
    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    plt.plot(gtposes[:,0],gtposes[:,1], linestyle='dashed',c='k')
    plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth','TartanVO'])
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)

def plot_trajs(gtposes, trajs, labels, vis=False, savefigname=None, figsize=(4,4)):
    colors= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bb3f3f', 'g','b','#c8fd3d','m','k','#0a481e','#db4bda']
    fig = plt.figure(figsize=figsize)
    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    plt.plot(gtposes[:,0],gtposes[:,1], linestyle='dashed',c='k',linewidth=2)
    for k,traj in enumerate(trajs):
        plt.plot(traj[:, 0], traj[:, 1],c=colors[k],linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth',] + labels, prop={'size': 13})
    # plt.legend(loc=2, prop={'size': 180})
    # plt.legend(fontsize=8)
    plt.title('KITTI 09')
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)

def evaluate_trajectory(gtposes, estposes, trajtype, outfilename, scale=False, medir_dir='./'):
    gtmotions = pose_quats2motion_ses(gtposes)
    estmotions = pose_quats2motion_ses(estposes)

    evaluator = TartanAirEvaluator('./')
    ate_score, rpe_score, kitti_score, files, ate_gt, ate_est = evaluator._evaluate_one_trajectory(gtposes, estposes, vid_concate=False,
                                                                scale=scale, save_files=False, aicrowd_submission_id='123', 
                                                                media_dir = medir_dir, medir_surfix='.mp4', vis_ate=True, kittitype=(trajtype=='kitti')) 
    # ate/rpe-r/rpe-t/kitti-r/kitti-t: 
    print('%.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (ate_score, rpe_score[0], rpe_score[1], kitti_score[0], kitti_score[1] ))

    return 

    est_norm = np.linalg.norm(estmotions[:,:3], axis=1)
    gt_norm = np.linalg.norm(gtmotions[:,:3], axis=1)
    costheta = np.sum((estmotions[:,:3] * gtmotions[:,:3]),axis=1)/(est_norm+1e-8)/(gt_norm+1e-8) # translation angle difference
    theta = np.arccos(costheta)
    scalediff = est_norm / (gt_norm+1e-6)
    motiondiff = estmotions - gtmotions

    fn2 = outfilename + '_viserror.jpg'
    plot_error_traj(theta, gtposes, estposes, motiondiff, vis=False, savefigname=fn2)
    # import ipdb;ipdb.set_trace()
    fn3 = outfilename + '_trajs.jpg'
    plot_traj(ate_gt, np.array(ate_est), vis=False, savefigname=fn3, figsize=(5,5))

    # save the scores
    scorefile = outfilename + '_score.txt'
    with open(scorefile, 'w') as f:
        f.write('ATE, RPE-R, PRE-T, KITTI-R, KITII-T\n')
        f.write('%.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (ate_score, rpe_score[0], rpe_score[1], kitti_score[0], kitti_score[1] ))

def kitti2tartan(traj):
    '''
    traj: in kitti style, N x 12 numpy array, in camera frame
    output: in TartanAir style, N x 7 numpy array, in NED frame
    '''
    T = np.array([[0,0,1,0],
                  [1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []

    for pose in traj:
        tt = np.eye(4)
        tt[:3,:] = pose.reshape(3,4)
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(SE2quat(ttt))
        
    return np.array(new_traj)


if __name__ == '__main__':
    
    trajno = '09'
    # gtfiles = ['data/kitti/kitti_09_pose_left.txt','data/kitti/kitti_10_pose_left.txt']
    gtfile = 'data/kitti/kitti_'+trajno+'_pose_left.txt'
    files = [#'trajtest/17_0_2_kitti_/17_0_kitti_09_flow_output_motion.txt',
             # 'trajtest/10_2_kitti_/10_2_kitti_10_flow_output_motion.txt',
             'trajtest/WO_LC'+trajno+'_motion_scale.txt',
             'trajtest/19_4_kitti_'+trajno+'_flow_output_motion.txt',
             # 'trajtest/17_4_kitti_/17_4_kitti_10_flow_output_motion.txt',
             'trajtest/17_6_kitti_/17_6_kitti_'+trajno+'_flow_output_motion.txt',
             ]
    scale = [False, True, True]

    gtposes = np.loadtxt(gtfile)#[4:,:]
    gtmotions = pose_quats2motion_ses(gtposes)

    estposes_list = []
    for k, file in enumerate(files):
        motions = np.loadtxt(file)
        poses = motion_ses2pose_quats(motions)
        if scale[k]:
            motions_scale = per_frame_scale_alignment(gtmotions, motions)
            motions_scale = np.array(motions_scale)
        else:
            # import ipdb;ipdb.set_trace()
            num = len(gtmotions) - len(motions)
            motions_scale = np.concatenate((gtmotions[:num,:],motions),axis=0)
        estposes = motion_ses2pose_quats(motions_scale)

        estposes_list.append(estposes)

    plot_trajs(gtposes, estposes_list, labels=['ORBSLAM','TartanVO','COMPASS'], vis=True, savefigname=None, figsize=(5,5)) #'Scrach', 

    # # import ipdb;ipdb.set_trace()
    # aaa= np.loadtxt('trajtest/WO_LC09.txt')
    # aaa=aaa[:,1:]
    # aaa=quats2SEs(aaa)
    # aaa=kitti2tartan(aaa)
    # bbb=pose_quats2motion_ses(aaa)
    # np.savetxt('trajtest/WO_LC09_motion.txt',bbb)

    # aaa=np.loadtxt('trajtest/WO_LC10_motion.txt')
    # bbb=motion_ses2pose_quats(aaa)
    # import ipdb;ipdb.set_trace()
    # from evaluator.evaluate_ate_scale import align_scale
    # num = len(gtposes) - len(bbb)
    # _, _, _, s = align_scale(gtposes[num:,:3].transpose(), bbb[:,:3].transpose(), True)
    # print(s)
    # bbb[:,:3] = bbb[:,:3] * s
    # ccc=pose_quats2motion_ses(bbb)
    # # ccc[:5,:] = per_frame_scale_alignment(gtmotions[:5,:], ccc[:5,:])
    # np.savetxt('trajtest/WO_LC10_motion_scale.txt',ccc)

