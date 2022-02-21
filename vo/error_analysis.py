import numpy as np
from evaluator.plot_trajectory import plot_gt_est_trajectories
from evaluator.tartanair_evaluator import TartanAirEvaluator
from evaluator.transformation import motion_ses2pose_quats, pose_quats2motion_ses
from evaluator.trajectory_transform import shift0
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import join

def get_color(cm, value, minv, maxv, inverse=False):
    value = (value-minv)/(maxv-minv)
    if inverse:
        value = 1-value
    return cm(value)

def plot_error_curve(theta, scalediff, motiondiff, vis=False, savefigname=None):

    fig = plt.figure(figsize=(16,10))
    plt.subplot(311)
    plt.plot(theta)
    plt.grid()
    plt.legend(['theta'])

    plt.subplot(312)
    plt.plot(scalediff)
    plt.grid()
    plt.legend(['translation scale'])

    plt.subplot(313)
    plt.plot(motiondiff[:,3])
    plt.plot(motiondiff[:,4])
    plt.plot(motiondiff[:,5])
    plt.grid()
    plt.legend(['x','y','z'])
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)

def plot_error_traj(theta, gtposes, estposes, motiondiff, vis=False, savefigname=None):
    fig = plt.figure(figsize=(16,8))
    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(121)
    plt.plot(gtposes[:,0],gtposes[:,1])
    for k in range(len(estposes)-1):
        error = motiondiff[k, 5] # yaw rotation? 
        c = get_color(cm, error, -0.006, 0.006) #-0.006, 0.006)
        plt.plot(estposes[k:k+2, 0], estposes[k:k+2, 1], color=c, linewidth=3)

    plt.subplot(122)
    plt.plot(gtposes[:,0],gtposes[:,1])
    for k in range(len(theta)-1):
        error = theta[k]
        c = get_color(cm, error, 0., 0.1, inverse=True) # 0.1
        plt.plot(estposes[k:k+2, 0], estposes[k:k+2, 1], color=c, linewidth=3)

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)

def plot_traj(gtposes, estposes, vis=False, savefigname=None, figsize=(4,4)):
    fig = plt.figure(figsize=figsize)
    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    plt.plot(gtposes[:,0],gtposes[:,1], linestyle='dashed',c='k')
    plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth', 'COMPASS'])
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)

def findfile(filelist, filestr):
    for ff in filelist:
        if filestr in ff:
            return ff
    return None

def evaluate_trajectory(gtposes, estposes, trajtype, outfilename, scale=False, medir_dir='./'):
    gtmotions = pose_quats2motion_ses(gtposes)
    estmotions = pose_quats2motion_ses(estposes)

    evaluator = TartanAirEvaluator('./')
    ate_score, rpe_score, kitti_score, files, ate_gt, ate_est = evaluator._evaluate_one_trajectory(gtposes, estposes, vid_concate=False,
                                                                scale=scale, save_files=False, aicrowd_submission_id='123', 
                                                                media_dir = medir_dir, medir_surfix='.mp4', vis_ate=True, kittitype=(trajtype=='kitti')) 
    # ate/rpe-r/rpe-t/kitti-r/kitti-t: 
    print("Error metrics:")
    print('ATE: %.4f \t RPE-R: %.4f \t RPE-t: %.4f \t r_rel: %.4f \t t_rel: %.4f' % (ate_score, rpe_score[0], rpe_score[1], kitti_score[0], kitti_score[1] ))

    est_norm = np.linalg.norm(estmotions[:,:3], axis=1)
    gt_norm = np.linalg.norm(gtmotions[:,:3], axis=1)
    costheta = np.sum((estmotions[:,:3] * gtmotions[:,:3]),axis=1)/(est_norm+1e-8)/(gt_norm+1e-8) # translation angle difference
    theta = np.arccos(costheta)
    scalediff = est_norm / (gt_norm+1e-6)
    motiondiff = estmotions - gtmotions

    fn = outfilename + '_error.jpg'
    plot_error_curve(theta, scalediff, motiondiff, vis=False, savefigname=fn) # vis=True, savefigname=None)#

    fn2 = outfilename + '_viserror.jpg'
    plot_error_traj(theta, gtposes, estposes, motiondiff, vis=False, savefigname=fn2)
    # import ipdb;ipdb.set_trace()
    fn3 = outfilename + '_trajs.jpg'
    plot_traj(ate_gt, np.array(ate_est), vis=False, savefigname=fn3, figsize=(6,6))
    fn3 = outfilename + '_trajs_aligned.jpg'
    plot_traj(gtposes, estposes, vis=False, savefigname=fn3, figsize=(6,6))

    # save the scores
    scorefile = outfilename + '_score.txt'
    with open(scorefile, 'w') as f:
        f.write('ATE, RPE-R, PRE-T, KITTI-R, KITII-T\n')
        f.write('%.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (ate_score, rpe_score[0], rpe_score[1], kitti_score[0], kitti_score[1] ))

if __name__ == '__main__':
    
    assert len(sys.argv)>1
    trajfolder = sys.argv[1]
    print(trajfolder)

    euroclist = ['MH_01_easy','MH_02_easy','MH_03_medium','MH_04_difficult','MH_05_difficult','V1_01_easy','V1_02_medium','V1_03_difficult','V2_01_easy','V2_02_medium','V2_03_difficult']
    euroclist = ['MH_04_difficult','MH_05_difficult','V1_02_medium','V1_03_difficult','V2_02_medium','V2_03_difficult']
    tartanlist = ['MH00','MH01','MH02','MH03','MH04','MH05','MH06','MH07','SH00','SH01','SH02','SH03','SH04','SH05','SH06','SH07']
    kittilist = ['00','01','02','03','04','05','06','07','08','09','10'] #
    realsenselist = ['01','02','03'] #

    if 'tartan' in trajfolder:
        trajlist = tartanlist
        trajtype = 'tartan'
        gtfiles = '/bigdata/tartantest/hard_testing/%s/pose_left.txt'
        gtmotionfiles = '/bigdata/tartantest/hard_testing/%s/motion.npy'
    elif 'euroc' in trajfolder:
        trajlist = euroclist
        trajtype = 'euroc'
        gtfiles = '/home/wenshan/tmp/data/EuRoC/%s_mav0_Undistorted/cam0/pose_left.txt'
        gtmotionfiles = '/home/wenshan/tmp/data/EuRoC/%s_mav0_Undistorted/cam0/motion.npy'
    elif 'kitti' in trajfolder:
        trajlist = kittilist
        trajtype = 'kitti'
        gtfiles = '/bigdata/tartanvo_data/kitti/vo/%s/pose_left.txt'
        gtmotionfiles = '/bigdata/tartanvo_data/kitti/vo/%s/motion.npy'
    elif 'realsense' in trajfolder:
        trajlist = realsenselist
        trajtype = 'realsense'
        gtfiles = '/home/wenshan/tmp/data/20200804/%s/Infra1/pose_left.txt' #'/home/wenshan/tmp/data/kitti/vo/%s/pose_left.txt'
        gtmotionfiles = '/home/wenshan/tmp/data/20200804/%s/Infra1/motion.npy'
    else:
        print('Unknown result folder..')
        exit()

    # import ipdb;ipdb.set_trace()
    # collect and organize the result files from the folder
    evaluator = TartanAirEvaluator('./')
    trajdir = 'trajtest/' + trajfolder
    allfiles = listdir(trajdir)
    scaled_pose_files = [ff for ff in allfiles if ff.endswith('pose_scale.txt')]
    scaled_motion_files = [ff for ff in allfiles if ff.endswith('motion_scale.npy')]
    pose_files = [ff for ff in allfiles if ff.endswith('pose.txt')]
    motion_files = [ff for ff in allfiles if ff.endswith('motion.npy')]
    scaled_pose_files.sort()
    scaled_motion_files.sort()
    pose_files.sort()
    motion_files.sort()

    for k, traj in enumerate(trajlist):
        est_posefile = scaled_pose_files[k]
        est_motionfile = scaled_motion_files[k]
        if not '_'+traj+'_' in est_posefile:
            est_posefile = findfile(scaled_pose_files, '_'+traj+'_')
        if not '_'+traj+'_' in est_motionfile:
            est_motionfile = findfile(scaled_motion_files, '_'+traj+'_')
        estfile = join(trajdir, est_posefile)
        estmotionfile = join(trajdir, est_motionfile)

        gtfile = gtfiles % (traj)
        gtmotionfile = gtmotionfiles % (traj)

        estposes = np.loadtxt(estfile)
        estmotions = np.load(estmotionfile)

        gtposes = np.loadtxt(gtfile)
        gtmotions = np.load(gtmotionfile)
        gtposes = shift0(gtposes)

        # 
        # plot_gt_est_trajectories([gtposes, estposes], showfig=True, savefigname=None, saveaniname=None)

        ate_score, rpe_score, kitti_score, files, ate_gt, ate_est = evaluator._evaluate_one_trajectory(gtfile, estfile, vid_concate=False,
                                                                    scale=False, save_files=False, aicrowd_submission_id='123', 
                                                                    media_dir = './', medir_surfix='.mp4', vis_ate=True, kittitype=(trajtype=='kitti')) 
        # ate/rpe-r/rpe-t/kitti-r/kitti-t: 
        print('%.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (ate_score, rpe_score[0], rpe_score[1], kitti_score[0], kitti_score[1] ))

        est_norm = np.linalg.norm(estmotions[:,:3], axis=1)
        gt_norm = np.linalg.norm(gtmotions[:,:3], axis=1)
        costheta = np.sum((estmotions[:,:3] * gtmotions[:,:3]),axis=1)/(est_norm+1e-8)/(gt_norm+1e-8)
        theta = np.arccos(costheta)
        motiondiff = estmotions - gtmotions

        # fn = estfile.replace('pose_scale.txt', 'scale_error.jpg')
        # plot_error_curve(theta, motiondiff,vis=False, savefigname=fn) # vis=True, savefigname=None)#

        # fn2 = estfile.replace('pose_scale.txt', 'scale_viserror.jpg')
        # plot_error_traj(theta, gtposes, estposes, motiondiff, vis=False, savefigname=fn2)
        # # import ipdb;ipdb.set_trace()
        fn3 = estfile.replace('pose_scale.txt', 'scale_traj_big.jpg')
        plot_traj(ate_gt, np.array(ate_est), vis=False, savefigname=fn3)