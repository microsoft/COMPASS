"""
Dataset utitlities.
"""

def normalize_v(v):
    """ Normalize velocity to [-1, 1]. 
        Ref: https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation/blob/e651be52ff8274c9f595e88b13fe42d51302403d/racing_utils/dataset_utils.py#L20 """
    # normalization of velocities from whatever to [-1, 1] range
    v_x_range = [-1, 7]
    v_y_range = [-3, 3]
    v_z_range = [-3, 3]
    v_yaw_range = [-1, 1]
    if len(v.shape) == 1:
        # means that it's a 1D vector of velocities
        v[0] = 2.0 * (v[0] - v_x_range[0]) / (v_x_range[1] - v_x_range[0]) - 1.0
        v[1] = 2.0 * (v[1] - v_y_range[0]) / (v_y_range[1] - v_y_range[0]) - 1.0
        v[2] = 2.0 * (v[2] - v_z_range[0]) / (v_z_range[1] - v_z_range[0]) - 1.0
        v[3] = 2.0 * (v[3] - v_yaw_range[0]) / (v_yaw_range[1] - v_yaw_range[0]) - 1.0
    elif len(v.shape) == 2:
        # means that it's a 2D vector of velocities
        v[:, 0] = 2.0 * (v[:, 0] - v_x_range[0]) / (v_x_range[1] - v_x_range[0]) - 1.0
        v[:, 1] = 2.0 * (v[:, 1] - v_y_range[0]) / (v_y_range[1] - v_y_range[0]) - 1.0
        v[:, 2] = 2.0 * (v[:, 2] - v_z_range[0]) / (v_z_range[1] - v_z_range[0]) - 1.0
        v[:, 3] = 2.0 * (v[:, 3] - v_yaw_range[0]) / (v_yaw_range[1] - v_yaw_range[0]) - 1.0
    else:
        raise Exception('Error in data format of V shape: {}'.format(v.shape))
    return v


def de_normalize_v(v):
    """ De-normalize velocity.
        Ref: https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation/blob/e651be52ff8274c9f595e88b13fe42d51302403d/racing_utils/dataset_utils.py#L42
    """
    # normalization of velocities from [-1, 1] range to whatever
    v_x_range = [-1, 7]
    v_y_range = [-3, 3]
    v_z_range = [-3, 3]
    v_yaw_range = [-1, 1]
    if len(v.shape) == 1:
        # means that it's a 1D vector of velocities
        v[0] = (v[0] + 1.0) / 2.0 * (v_x_range[1] - v_x_range[0]) + v_x_range[0]
        v[1] = (v[1] + 1.0) / 2.0 * (v_y_range[1] - v_y_range[0]) + v_y_range[0]
        v[2] = (v[2] + 1.0) / 2.0 * (v_z_range[1] - v_z_range[0]) + v_z_range[0]
        v[3] = (v[3] + 1.0) / 2.0 * (v_yaw_range[1] - v_yaw_range[0]) + v_yaw_range[0]
    elif len(v.shape) == 2:
        # means that it's a 2D vector of velocities
        v[:, 0] = (v[:, 0] + 1.0) / 2.0 * (v_x_range[1] - v_x_range[0]) + v_x_range[0]
        v[:, 1] = (v[:, 1] + 1.0) / 2.0 * (v_y_range[1] - v_y_range[0]) + v_y_range[0]
        v[:, 2] = (v[:, 2] + 1.0) / 2.0 * (v_z_range[1] - v_z_range[0]) + v_z_range[0]
        v[:, 3] = (v[:, 3] + 1.0) / 2.0 * (v_yaw_range[1] - v_yaw_range[0]) + v_yaw_range[0]
    else:
        raise Exception('Error in data format of V shape: {}'.format(v.shape))
    return v


def remove_prefix(text, prefix):
    """ 
    Remove prefix of a string.
    Ref: https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever