from .resnet_2d3d import * 
from .resnet_2d3d import resnet18_2d3d_full_C1, resnet18_2d3d_full_C2

def select_resnet(network, norm='bn'):
    param = {'feature_size': 1024}
    #backbones = {}
    if network == 'resnet18':
        model_img = resnet18_2d3d_full(track_running_stats=True, norm=norm)
        model_flow = resnet18_2d3d_full_C2(track_running_stats=True, norm=norm) 
        model_seg = resnet18_2d3d_full_C1(track_running_stats=True, norm=norm)
        model_depth = resnet18_2d3d_full_C1(track_running_stats=True, norm=norm)

        param['feature_size'] = 256
    elif network == 'resnet34':
        model = resnet34_2d3d_full(track_running_stats=True, norm=norm)
        param['feature_size'] = 256 
    elif network == 'resnet50':
        model = resnet50_2d3d_full(track_running_stats=True, norm=norm)
    else: 
        raise NotImplementedError

    return model_img, model_seg, model_depth, model_flow,  param
    #return backbones, param
