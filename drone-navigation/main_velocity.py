import argparse
import os
import random
import time
import warnings

import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from datasets import ILDataset, ILVideoDataset
from models.compass.compass_model import CompassModel
from utils.dataset_utils import de_normalize_v, remove_prefix
from utils.meters import AverageMeter, ProgressMeter
from utils import logger

# Original arguments from ImageNet example.
model_names = [
    'compass'
    ]
parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='compass',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: compass)')
parser.add_argument('--arch_settings', default='{}',
                    help='architecture settings (default: "{}")')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run (default: 15')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# New arguments.
parser.add_argument('--output_dir', default='output',
                    help="Output directory.")
parser.add_argument('--dataset_dir', default='data/il',
                    help='Dataset directory.')
parser.add_argument('--train_ann_file_name', default='bc_v5_n0_train_ann.txt',
                    help='Training annotation file name (default: bc_v5_n0_train_ann.txt).')
parser.add_argument('--val_ann_file_name', default='bc_v5_n0_val_ann.txt',
                    help='Validation annotation file name (default: bc_v5_n0_val_ann.txt).')
parser.add_argument('--pretrained_encoder_path', type=str, default='',
                    help='Path of pre-trained encoder.')
parser.add_argument('--pretrained_encoder_type', type=str, default='',
                    help='Type of pre-trained encoder.')
parser.add_argument('--opt', type=str, default='Adam',
                    help='Optimizer type.')
parser.add_argument('--wd', default=1e-3, type=float, 
                    help= 'weight decay')
parser.add_argument('--save_freq', default=5, type=int,
                    help='Frequency of checkpoint saving in epochs (default: 5).')
parser.add_argument('--disable_random_resize_crop', action='store_true',
                    help='Use Resize instead of RandomResizeCrop.')
parser.add_argument('--freeze_encoder', action='store_true',
                    help='Freeze weights in encoder.')
parser.add_argument('--evaluate_models', action='store_true',
                    help='evaluate multiple models on validation set')
parser.add_argument('--resume_dir', type=str, default='output',
                    help='(for evaluating models) directory of the models to resume (default: output)')
parser.add_argument('--resume_epochs', type=str, default='0,4,9,14',
                    help='(for evaluating models) directory of the models to resume (default: 0,4,9,14)')
parser.add_argument('--data_type', default='image',
                    help='Data type (default: image)')
parser.add_argument('--clip_len', default=8, type=int,
                    help='Number of frames in each video clip (default: 8).')
parser.add_argument('--use_checkpoint', action='store_true',
                    help='Use checkpointing method to save memory.')
parser.add_argument('--patch_size', type=str, default='(4,4,4)',
                    help='Patch size for video Swin model (default (4,4,4)).')
parser.add_argument('--window_size', type=str, default='(8,7,7)',
                    help='Window size for video Swin model (default (8,7,7)).')
parser.add_argument('--scheduler', type=str, default='',
                    help='Learning rate scheduler (default "").')
parser.add_argument('--zip_file_name', type=str, default='',
                    help='Zipped dataset file name instead of dataset folder (default "").')
parser.add_argument('--flow_type', type=str, default='',
                    help='Flow encoder type for additional features (default "").')
parser.add_argument('--flow_normalizer', type=float, default=1.0,
                    help='Normalizer for flow output (default 1.0).')
parser.add_argument('--use_flow_vis_as_img', action='store_true',
                    help='Use flow visualization as input image.')
parser.add_argument('--use_depth_vis_as_img', action='store_true',
                    help='Use depth visualization as input image.')
parser.add_argument('--cor_feat_indices', type=str, default='[]',
                    help='Correspondence feature indices (default "").')
parser.add_argument('--cor_feat_names', type=str, default='[]',
                    help='Correspondence feature names (default "").')
parser.add_argument('--linear_prob', action='store_true',
                    help='linear prediction or maintain spatial info')
parser.add_argument('--use_gru', action='store_true',
                    help='load pretrained weights of Bidirectional GPU.')
parser.add_argument('--use_memory', action='store_true')

def main():
    args = parser.parse_args()
    print('Arguments:', args)

    # Create the output dir.
    print('Create output dir:', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.setup(os.path.join(args.output_dir, "logs"))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function.
        assert args.gpu is not None                      # Only allow single GPU mode. DataParallel mode is disabled.
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # Note: args.gpu is overwritten by gpu.
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
      
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'compass':
        model = CompassModel(args)
    else:
        raise ValueError()

    # freeze encoder's weights if needed
    if args.freeze_encoder:
        print("=> freeze encoder's weights")
        for param in model.encoder.parameters():
            param.requires_grad = False  # not update by gradient
        
    # set different lr ####
    print("=> trainable parameters:")
    params = []
    for name, param in model.named_parameters():
        if ('encoder' in name) or ('agg_f' in name) or ('agg_b' in name):
            params.append({'params': param, 'lr': args.lr})
            #print('-', name, param.shape)
        else:
            params.append({'params': param})
    # ====== Check Grad===================
    print("=> trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("- ", name, param.shape)
            continue
    

    # set GPU and migrate model
    if not torch.cuda.is_available():
        # Only allow GPU mode.
        raise RuntimeError("Requires GPU access. CPU only not supported")        
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss(reduction="mean")
    criterion = criterion.cuda(args.gpu)

    if params is None: params = model.parameters()
    if args.opt == 'Adam':
        #optimizer = torch.optim.Adam(model.parameters(), args.lr)
        optimizer = torch.optim.Adam(params, args.lr) 
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    else:
        raise ValueError()

    if args.scheduler == 'MultiStepLR':
        # Ref: https://pytorch.org/docs/stable/optim.html
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8], gamma=0.1)
    elif args.scheduler == '':
        scheduler = None
    else:
        raise ValueError()

    cudnn.benchmark = True

    # data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.disable_random_resize_crop:  
        train_transform = transforms.Compose([
            transforms.Resize(224),             # Only resize.
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # Note: RandomResizedCrop() will find a random crop first, then resize. Sometimes the original gate disappears!
            transforms.ToTensor(),
            normalize
        ])

    val_transform = transforms.Compose([
        transforms.Resize(224),  # Note: It is different from the ImageNet example, which resizes image to 256x256,
        transforms.ToTensor(),   #       and then center crops it to 224x224.
        normalize
    ])

    if args.data_type == 'image':
        train_dataset = ILDataset(dataset_dir=args.dataset_dir, ann_file_name=args.train_ann_file_name, transform=train_transform)
        val_dataset = ILDataset(dataset_dir=args.dataset_dir, ann_file_name=args.val_ann_file_name, transform=val_transform)
    elif args.data_type == 'video':
        train_dataset = ILVideoDataset(
            dataset_dir=args.dataset_dir, ann_file_name=args.train_ann_file_name, zip_file_name=args.zip_file_name, 
            transform=train_transform, clip_len=args.clip_len, use_flow=(args.flow_type != ''), use_flow_vis_as_img=args.use_flow_vis_as_img,
            use_depth_vis_as_img=args.use_depth_vis_as_img)
        val_dataset = ILVideoDataset(
            dataset_dir=args.dataset_dir, ann_file_name=args.val_ann_file_name, zip_file_name=args.zip_file_name, 
            transform=val_transform, clip_len=args.clip_len, use_flow=(args.flow_type != ''), use_flow_vis_as_img=args.use_flow_vis_as_img,
            use_depth_vis_as_img=args.use_depth_vis_as_img)
    else:
        raise ValueError()
    print(f'Training set: {len(train_dataset)}, val set: {len(val_dataset)}.')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    def resume_for_eval(model, path, gpu):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            if gpu is None:
                checkpoint = torch.load(path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(path, map_location=loc)
            if hasattr(model, 'module'):
                model_to_load = model.module
            else:
                model_to_load = model
            state_dict = {remove_prefix(k, 'module.'): v for k, v in checkpoint['state_dict'].items()}
            model_to_load.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {}, starts from 1)"
                  .format(path, checkpoint['epoch']))
            epoch = checkpoint['epoch'] - 1   # Epoch that starts from 0, which is used during training.
            return epoch
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(path))

    # Evaluate single model.
    if args.evaluate:
        # Must resume from checkpoint for evaluation.
        epoch = resume_for_eval(model, args.resume, args.gpu)
        validate(val_loader, model, criterion, epoch, args)
        return

    # Evaluate multiple models.
    if args.evaluate_models:
        # Must resume from checkpoint for evaluation.
        resume_epochs = [int(epoch) for epoch in args.resume_epochs.split(',')]
        for epoch in resume_epochs:
            path = os.path.join(args.resume_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
            epoch = resume_for_eval(model, path, args.gpu)
            validate(val_loader, model, criterion, epoch, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # eval on validation set
        validate(val_loader, model, criterion, epoch, args)

        # update learning rate if needed
        if scheduler:
            print(f'Take a step for scheduler at epoch: {epoch}.')  # FIXME: Does it really work?
            scheduler.step()

        # Note: We use args.rank == 0 instead of args.rank % ngpus_per_node == 0 to avoid duplicate checkpoint writing.
        # FIXME: This condition has potential issue: When args.distributed is True but args.multiprocessing_distributed is False,
        #        each instance will write checkpoint, which may cause conflict if shared storage is used.
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0):  
            if (epoch + 1) % args.save_freq == 0 or epoch < 1:
                print('save_checkpoint - args.output_dir: {}'.format(args.output_dir))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename='checkpoint_{:04d}.pth.tar'.format(epoch), output_dir=args.output_dir)

        logger.dumpkvs()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    errors = AverageMeter('Error', '')  # For errors, we use {} format for NumPy array.
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, samples in enumerate(train_loader):
        imgs, vels = samples['img'], samples['vel']
        if args.flow_type:
            flows = samples['flow']

        # measure data loading time
        data_time.update(time.time() - end)

        imgs = imgs.cuda(args.gpu, non_blocking=True)
        vels = vels.cuda(args.gpu, non_blocking=True)
        if args.flow_type:
            flows = flows.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.flow_type:
            output = model(imgs, flows)
        else:
            output = model(imgs)
        loss = criterion(output, vels)
        
        # measure error
        denorm_output = de_normalize_v(output.detach().cpu().numpy())
        denorm_vels = de_normalize_v(vels.detach().cpu().numpy())
        error = np.absolute(denorm_output - denorm_vels).mean(axis=0)  # Average L1 error over mini-batch. Shape: [4,].

        losses.update(loss.item(), imgs.size(0))
        errors.update(error, imgs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    logger.logkv("train/loss", losses.avg)
    for i, v in enumerate(errors.avg):
        logger.logkv(f"train/e_{i}", v)
    print('Train: epoch {}, avg loss {}, avg error {}'.format(epoch, losses.avg, errors.avg))

    
def validate(val_loader, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    errors = AverageMeter('Error', '')  # For errors, we use {} format for NumPy array.

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        all_denorm_output = []
        all_denorm_vels = []

        for _, samples in enumerate(val_loader):
            imgs, vels = samples['img'], samples['vel']
            if args.flow_type:
                flows = samples['flow']

            # always move to gpu
            imgs = imgs.cuda(args.gpu, non_blocking=True)
            vels = vels.cuda(args.gpu, non_blocking=True)
            if args.flow_type:
                flows = flows.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.flow_type:
                output = model(imgs, flows)
            else:
                output = model(imgs)
            loss = criterion(output, vels)
    
            # measure error
            denorm_output = de_normalize_v(output.detach().cpu().numpy())
            denorm_vels = de_normalize_v(vels.detach().cpu().numpy())
            error = np.absolute(denorm_output - denorm_vels).mean(axis=0)  # Average L1 error over mini-batch. Shape: [4,].

            losses.update(loss.item(), imgs.size(0))
            errors.update(error, imgs.size(0))
            all_denorm_output.append(denorm_output)
            all_denorm_vels.append(denorm_vels)

        print('Val: epoch {}, avg loss {}, avg error {}'.format(epoch, losses.avg, errors.avg))

        logger.logkv("valid/loss", losses.avg)
        for i, v in enumerate(errors.avg):
            logger.logkv(f"valid/e_{i}", v)

        if (not args.distributed) or (args.distributed and args.rank == 0):
            all_denorm_output = np.concatenate(all_denorm_output, axis=0)
            all_denorm_vels = np.concatenate(all_denorm_vels, axis=0)
            all_preds = {'all_denorm_output': all_denorm_output, 'all_denorm_vels': all_denorm_vels}
            torch.save(all_preds, os.path.join(args.output_dir, f'all_preds_epoch{epoch}.pth'))
        

def save_checkpoint(state, filename, output_dir):
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(state, checkpoint_path)


if __name__ == '__main__':
    main()