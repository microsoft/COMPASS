import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CompassModel(nn.Module):
    def __init__(self, args):
        super(CompassModel, self).__init__()

        self.args = args
        from .select_backbone import select_resnet
        self.encoder, _, _, _, param = select_resnet('resnet18')

        if args.linear_prob:
            self.pred = nn.Sequential(
                nn.Linear(param['feature_size'], 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1)
            )
        else:
            self.pred = nn.Conv2d(param['feature_size'], param['feature_size'], kernel_size=1, padding=0)
 
        self._initialize_weights(self.pred)
        self.load_pretrained_encoder_weights(args.pretrained_encoder_path)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 0.1)

    def load_pretrained_encoder_weights(self, pretrained_path):
        if pretrained_path:
            ckpt = torch.load(pretrained_path)['state_dict']  # COMPASS checkpoint format.
            ckpt2 = {}
            for key in ckpt:
                if key.startswith('backbone_rgb'):
                    ckpt2[key.replace('backbone_rgb.', '')] = ckpt[key]
                elif key.startswith('module.backbone'):
                    ckpt2[key.replace('module.backbone.', '')] = ckpt[key]
            self.encoder.load_state_dict(ckpt2)
            print('Successfully loaded pretrained checkpoint: {}.'.format(pretrained_path))
        else:
            print('Train from scratch.')
    
    def forward(self, x):
        # x: B, C, SL, H, W
        #x = x.unsqueeze(2)           # Shape: [B,C,H,W] -> [B,C,1,H,W].
        x = self.encoder(x)          # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.

        if self.args.linear_prob:
            x = x.mean(dim=(2, 3, 4))    # Shape: [B,C',1,H',W'] -> [B,C'].
            x = self.pred(x)             # Shape: [B,C'] -> [B,C''].
            
        else:
            #TODO
            print('using convd')
            B, N, T, H, W = x.shape
            x = x.view(B, T, N, H, W)
            x = x.view(B*T, N, H, W)
            x = self.pred(x) 
            x = x.mean(dim=(1, 2, 3))
        return x