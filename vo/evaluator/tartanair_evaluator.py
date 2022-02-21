import numpy as np
from .evaluator_base import ATEEvaluator, RPEEvaluator, KittiEvaluator, transform_trajs, quats2SEs
from .plot_trajectory import plot_gt_est_trajectories, animate_gt_est_trajectories
from zipfile import ZipFile
from os import listdir, system
from os.path import isdir, isfile

# from trajectory_transform import timestamp_associate


def concatenate_videos(vid1, vid2, outfile):
    stringa = "ffmpeg -i \"concat:"
    elenco_video = [vid1, vid2]
    elenco_file_temp = []
    for f in elenco_video:
        file = "temp" + str(elenco_video.index(f) + 1) + ".ts"
        if isfile(file):
            system('rm '+file)
        system("ffmpeg -i " + f + " -c copy -bsf:v h264_mp4toannexb -f mpegts " + file)
        elenco_file_temp.append(file)
    print(elenco_file_temp)
    for f in elenco_file_temp:
        stringa += f
        if elenco_file_temp.index(f) != len(elenco_file_temp)-1:
            stringa += "|"
        else:
            stringa += "\" -c copy  -bsf:a aac_adtstoasc " + outfile
    print(stringa)
    system(stringa)
 

class TartanAirEvaluator:
    def __init__(self, answer_file_path, mono_track = True, round=1):
        """
        `round` : Holds the round for which the evaluation is being done. 
        can be 1, 2...upto the number of rounds the challenge has.
        Different rounds will mostly have different ground truth files.
        """
        self.answer_file_path = answer_file_path

        # monocular track
        self.scale = mono_track

        # # stereo track
        # self.scale = False

        self.ate_eval = ATEEvaluator()
        self.rpe_eval = RPEEvaluator()
        self.kitti_eval = KittiEvaluator()

        # score balencing factor
        self.beta = 7.0
        
    def _evaluate_one_trajectory(self, gt_traj, est_traj, scale=False, save_files=False, vid_concate=True, aicrowd_submission_id='', 
                                    media_dir = '/tmp/', medir_surfix='.mp4', vis_ate=False, kittitype=True):

        # TODO: try catch exceptions
        # load trajectories
        # import ipdb;ipdb.set_trace()
        try:
            gt_traj = np.loadtxt(gt_traj)
            est_traj = np.loadtxt(est_traj)
        except:
            pass
        if gt_traj.shape[0] != est_traj.shape[0]:
            raise Exception("POSEFILE_LENGTH_ILLEGAL")
        if gt_traj.shape[1] != 7 or est_traj.shape[1] != 7:
            raise Exception("POSEFILE_FORMAT_ILLEGAL")

        # transform and scale
        gt_traj_trans, est_traj_trans, s = transform_trajs(gt_traj, est_traj, scale)
        gt_SEs, est_SEs = quats2SEs(gt_traj_trans, est_traj_trans)

        ate_score, gt_ate_aligned, est_ate_aligned = self.ate_eval.evaluate(gt_traj, est_traj, scale)
        rpe_score = self.rpe_eval.evaluate(gt_SEs, est_SEs)
        kitti_score = self.kitti_eval.evaluate(gt_SEs, est_SEs, kittitype=kittitype)

        files = {}
        if save_files:
            if vis_ate:
                # align ate est with rpe gt
                ate_figure_name = media_dir+str(aicrowd_submission_id)+'_ate.jpg'
                ate_animation_name = media_dir+str(aicrowd_submission_id)+'_ate.mp4'
                plot_gt_est_trajectories([gt_ate_aligned, est_ate_aligned], showfig=False, savefigname=ate_figure_name, saveaniname=ate_animation_name)
            kitti_figure_name = media_dir+str(aicrowd_submission_id)+'_kitti.jpg'
            kitti_animation_name = media_dir+str(aicrowd_submission_id)+'_kitti'+medir_surfix
            kitti_gen_animation_name = media_dir+str(aicrowd_submission_id)+'_kitti_gen'+medir_surfix
            plot_gt_est_trajectories([gt_traj_trans, est_traj_trans], showfig=False, savefigname=kitti_figure_name, saveaniname=kitti_animation_name)
            animate_gt_est_trajectories([gt_traj_trans, est_traj_trans], saveaniname=kitti_gen_animation_name)

            if vid_concate:
                # combine two mp4 files into one
                combined_animation_name = media_dir+str(aicrowd_submission_id) + medir_surfix
                concatenate_videos(kitti_animation_name, kitti_gen_animation_name, combined_animation_name)
                files["media_video_path"] = combined_animation_name
            else:
                files["media_video_path"] = kitti_animation_name

            files["media_video_thumb_path"] = kitti_animation_name

        return ate_score, rpe_score, kitti_score, files, gt_ate_aligned, est_ate_aligned

    def _evaluate(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
            - submission_file_path : local file path of the submitted file
            - aicrowd_submission_id : A unique id representing the submission
            - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]
        aicrowd_submission_id = client_payload["aicrowd_submission_id"]
        aicrowd_participant_uid = client_payload["aicrowd_participant_id"]
        
        # read zip file
        if isfile(submission_file_path) and submission_file_path[-3:]=='zip':
            submit_zipfile = ZipFile(submission_file_path)
            submit_file_list = submit_zipfile.namelist()
        else:
            raise Exception("UNKNOWN SUBMISSION FILE {}".format(submission_file_path))

        # answer_file_path is a directory stores all the gt files
        if isdir(self.answer_file_path): 
            file_list = listdir(self.answer_file_path)
        else:
            raise Exception("CANNOT FIND GT FILES in {}".format(self.answer_file_path))
        file_list = [ff for ff in file_list if ff[-3:]=='txt']
        file_list.sort()

        # evaluate the trajectories one by one
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        kitti_trans_list = []
        kitti_rot_list = []
        _result_object = {}
        for k, filename in enumerate(file_list):
            gt_filename = self.answer_file_path + '/' + filename
            if filename in submit_file_list:
                print('Evaluate file {}'.format(filename))
                est_file = submit_zipfile.open(filename)
                # only save visualization file for the first trajectory
                ate_score, rpe_score, kitti_score, files = self._evaluate_one_trajectory(gt_filename, est_file, 
                            scale=self.scale, save_files=(k==0), aicrowd_submission_id=aicrowd_submission_id) 
                print('  scores: {}, {}, {}'.format(ate_score, rpe_score, kitti_score))
                ate_list.append(ate_score)
                rpe_trans_list.append(rpe_score[1])
                rpe_rot_list.append(rpe_score[0])
                kitti_trans_list.append(kitti_score[1])
                kitti_rot_list.append(kitti_score[0])
                if len(files)>0: # update the output file path
                    for kk in files.keys():
                        _result_object[kk] = files[kk]
            else: # missing file
                raise Exception("MISSING SUBMISSION FILE {}".format(filename))

        ate_mean = np.array(ate_list).mean()
        rpe_trans_mean = np.array(rpe_trans_list).mean()
        rpe_rot_mean = np.array(rpe_rot_list).mean()
        kitti_trans_mean = np.array(kitti_trans_list).mean()
        kitti_rot_mean = np.array(kitti_rot_list).mean()

        _result_object['score'] = kitti_rot_mean + kitti_trans_mean * self.beta
        _result_object['score_secondary'] = ate_mean
        _result_object['private'] = { 'rpe_translation': rpe_trans_mean, 
                                      'rpe_rotation': rpe_rot_mean, 
                                      'kitti_translation': kitti_trans_mean, 
                                      'kitti_rotation': kitti_rot_mean,
                                      'ate': ate_mean
        }
        return _result_object

if __name__ == "__main__":
    answer_file_path = "data/mono_gt" #'data/toy_gt2.txt' #
    #answer_file_path = "data/stereo_gt" #'data/toy_gt2.txt' #
    _client_payload = {}
    _client_payload["submission_file_path"] = "data/mono_est.zip" # 'data/toy_est2.txt' # 
    #_client_payload["submission_file_path"] = "data/stereo_est.zip" # 'data/toy_est2.txt' # 
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234
    
    # Instaiate a dummy context
    _context = {}
    # scale = True for monocular track, scale = False for stereo track
    aicrowd_evaluator = TartanAirEvaluator(answer_file_path, mono_track=True)
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)

    # gt_traj_name = 'data/stereo_gt/SH000.txt'
    # est_traj_name = 'data/stereo_est/SH000.txt'

    # result = aicrowd_evaluator._evaluate_one_trajectory(gt_traj_name, est_traj_name, scale=False, save_files=False, vid_concate=False, aicrowd_submission_id='', media_dir = '/tmp/')
    # print(result)
