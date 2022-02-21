# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--working-dir', default='./',
                        help='working directory')

    parser.add_argument('--exp-prefix', default='debug_',
                        help='exp prefix')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 3e-4)')

    parser.add_argument('--lr-decay', action='store_true', default=False,
                        help='decay of learning rate.')

    parser.add_argument('--normalize-output', type=float, default=0.05,
                        help='normalize the output (default: 1)')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')

    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')

    parser.add_argument('--train-step', type=int, default=1000000,
                        help='number of interactions in total (default: 1000000)')

    parser.add_argument('--snapshot', type=int, default=2000,
                        help='snapshot (default: 100000)')

    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')

    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 480)')

    parser.add_argument('--hsv-rand', type=float, default=0.0,
                        help='augment rand-hsv by adding different hsv to a set of images (default: 0.0)')

    parser.add_argument('--data-file', default='',
                        help='txt file specify the training data (default: "")')
    parser.add_argument('--data-npy', default='',
                        help='txt file specify the validation data (default: "")')
   


    parser.add_argument('--val-file', default='',
                        help='txt file specify the validation data (default: "")')
    parser.add_argument('--val-npy', default='',
                        help='txt file specify the validation data (default: "")')

    parser.add_argument('--train-data-type', default='tartan',
                        help='sceneflow / kitti / tartan')

    parser.add_argument('--train-data-balence', default='1',
                        help='sceneflow / kitti / tartan')

    parser.add_argument('--test-data-type', default='tartan',
                        help='sceneflow / kitti / tartan')

    parser.add_argument('--load-model', action='store_true', default=False,
                        help='load pretrained model (default: False)')

    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')

    parser.add_argument('--test', action='store_true', default=False,
                        help='test (default: False)')

    parser.add_argument('--test-num', type=int, default=10,
                        help='test (default: 10)')

    parser.add_argument('--test-save-image', action='store_true', default=False,
                        help='test output image to ./testimg (default: False)')

    parser.add_argument('--test-interval', type=int, default=100,
                        help='The test interval.')

    parser.add_argument('--use-int-plotter', action='store_true', default=False,
                        help='Enable cluster mode.')

    parser.add_argument('--print-interval', type=int, default=10,
                        help='The plot interval for updating the figures.')

    parser.add_argument('--plot-interval', type=int, default=100000000000000,
                        help='The plot interval for updating the figures.')

    parser.add_argument('--no-data-augment', action='store_true', default=False,
                        help='no data augmentation (default: False)')

    parser.add_argument('--multi-gpu', type=int, default=1,
                        help='multiple gpus numbers (default: False)')

    parser.add_argument('--platform', default='local',
                        help='deal with different data root directory in dataloader, could be one of local, cluster, azure (default: "local")')

    # VO-Flow
    parser.add_argument('--downscale-flow', action='store_true', default=False,
                        help='when resvo, use 1/4 flow size, which is the size output by pwc')

    parser.add_argument('--azure', action='store_true', default=False,
                        help='(deprecate - platform) training on azure (default: False)')

    parser.add_argument('--load-from-e2e', action='store_true', default=False,
                        help='load pose model from end2end flow-pose model')

    parser.add_argument('--test-traj-dir', default='',
                        help='test trajectory folder for flowvo (default: "")')

    parser.add_argument('--test-output-dir', default='./',
                        help='output dir of the posefile and media files for flowvo (default: "")')

    parser.add_argument('--traj-pose-file', default='',
                        help='test trajectory gt pose file (default: "")')
   
    parser.add_argument('--test-worker-num', type=int, default=1,
                        help='data loader worker number for testing set (default: 10)')

    parser.add_argument('--test-file-name', default='',
                        help='save video file name when testing')

    parser.add_argument('--intrinsic-layer', action='store_true', default=False,
                        help='add two layers as intrinsic input')

    parser.add_argument('--random-crop', type=int, default=0,
                        help='crop and resize the flow w/ intrinsic layers')

    parser.add_argument('--random-crop-center', action='store_true', default=False,
                        help='random crop at the center of the image')

    parser.add_argument('--random-intrinsic', type=float, default=0,
                        help='similar with random-crop but cover contineous intrinsic values')

    parser.add_argument('--fix-ratio', action='store_true', default=False,
                        help='fix resize ratio')

    parser.add_argument('--load-flow-model', action='store_true', default=False,
                        help='In end2end training, load pretrained flow model')

    parser.add_argument('--flow-model', default='pwc_net.pth.tar',
                        help='In end2end training, the name of pretrained flow model')

    parser.add_argument('--load-pose-model', action='store_true', default=False,
                        help='In end2end training, load pretrained pose model')

    parser.add_argument('--pose-model', default='',
                        help='In end2end training, the name of pretrained pose model')

    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')

    parser.add_argument('--realsense', action='store_true', default=False,
                        help='realsense test (default: False)')

    parser.add_argument('--test-traj', action='store_true', default=False,
                        help='test trajectory from --test-traj-dir (default: False)')

    parser.add_argument('--save-trajs', action='store_true', default=False,
                        help='save trajectories as animation files (default: False)')

    parser.add_argument('--scale-w', type=float, default=1.0,
                        help='scale_w for the kitti transform')

    parser.add_argument('--no-gt-motion', action='store_true', default=False,
                        help='test wo/ go motion (default: False)')

    parser.add_argument('--flow-thresh', type=float, default=10.0,
                        help='in end2end training, skip the sample if the flow error is bigger than thresh (default: 10.0)')

    parser.add_argument('--flow-folder', default='flow',
                        help='for train_voflow_wf.py, specify the flow folder when testing trajs')

    parser.add_argument('--train-flow', action='store_true', default=False,
                        help='step 1, only train flow (default: False)')

    parser.add_argument('--train-vo', action='store_true', default=False,
                        help='step 2, no extra flow and train vo e2e (default: False)')

    parser.add_argument('--fix-flow', action='store_true', default=False,
                        help='step 2, when train-vo=Ture, fix flownet and train vo (default: False)')

    parser.add_argument('--lambda-flow', type=float, default=1.0,
                        help='lambda vo in the loss function (default: 1.0)')

    parser.add_argument('--flow-file', default='',
                        help='txt file specify the training data (default: "")')

    parser.add_argument('--flow-data-balence', default='1',
                        help='specify the data balence scale e.g. 1,5,10 ')

    parser.add_argument('--flow-data-type', default='tartan',
                        help='sintel / flying / tartan')

    parser.add_argument('--lr-flow', type=float, default=1e-4,
                        help='learning rate for flow in e2e training (default: 1e-4)')

    parser.add_argument('--network', type=int, default=0,
                        help='network')

    parser.add_argument('--no-gt', action='store_true', default=False,
                        help='test wo/ gt motion/disp/flow (default: False)')

    parser.add_argument('--gt-pose-file', default='',
                        help='trajectory GT pose file using for visualization when testing )')


    parser.add_argument('--vo-gt-flow', type=float, default=0.0,
                        help='when e2e, use GT flow instead of e2e with this probability')

    parser.add_argument('--compressed', action='store_true', default=False, 
        help="Load the data that has been compressed. ")
    

    parser.add_argument('--pretrain-model-name', default='',
                        help='name of the pretrain model (default: "")')

    parser.add_argument('--use-gru', action='store_true', default=False, 
        help="Use GRU feature of the pretrain model. ")

    parser.add_argument('--resize-no-crop', action='store_true', default=False, 
        help="Resize the input to a target size. ")

    parser.add_argument('--pretrain-lr-scale', type=float, default=1.0,
                        help='scale the lr of pretrained model')

    parser.add_argument('--pre-train', action='store_true', default=False)
    parser.add_argument('--work_dir', default='./tmp', type=str)
    parser.add_argument('--save_tb', default=False)

    args = parser.parse_args()

    return args
