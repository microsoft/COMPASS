"""
Copyright (c) 2014, Alexander Fabisch
https://github.com/rock-learning/pytransform3d
"""
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import artist
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D, axes3d
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D

class Frame(artist.Artist):
    """A Matplotlib artist that displays a frame represented by its basis.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    label : str, optional (default: None)
        Name of the frame

    s : float, optional (default: 1)
        Length of basis vectors

    Other arguments except 'c' and 'color' are passed onto Line3D.
    """
    def __init__(self, A2B, label=None, s=1.0, **kwargs):
        super(Frame, self).__init__()

        if "c" in kwargs:
            kwargs.pop("c")
        if "color" in kwargs:
            kwargs.pop("color")

        self.s = s

        self.x_axis = Line3D([], [], [], color="r", **kwargs)
        self.y_axis = Line3D([], [], [], color="g", **kwargs)
        self.z_axis = Line3D([], [], [], color="b", **kwargs)

        self.draw_label = label is not None
        self.label = label

        if self.draw_label:
            self.label_indicator = Line3D([], [], [], color="k", **kwargs)
            self.label_text = Text3D(0, 0, 0, text="", zdir="z")

        self.set_data(A2B, label)

    def set_data(self, A2B, label=None):
        """Set the transformation data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        label : str, optional (default: None)
            Name of the frame
        """
        R = A2B[:3, :3]
        p = A2B[:3, 3]

        for d, b in enumerate([self.x_axis, self.y_axis, self.z_axis]):
            b.set_data([p[0], p[0] + self.s * R[0, d]],
                       [p[1], p[1] + self.s * R[1, d]])
            b.set_3d_properties([p[2], p[2] + self.s * R[2, d]])

        if self.draw_label:
            if label is None:
                label = self.label
            label_pos = p + 0.5 * self.s * (R[:, 0] + R[:, 1] + R[:, 2])

            self.label_indicator.set_data(
                [p[0], label_pos[0]], [p[1], label_pos[1]])
            self.label_indicator.set_3d_properties([p[2], label_pos[2]])

            self.label_text.set_text(label)
            self.label_text.set_position([label_pos[0], label_pos[1]])
            self.label_text.set_3d_properties(label_pos[2])


    @artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        """Draw the artist."""
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            b.draw(renderer, *args, **kwargs)
        if self.draw_label:
            self.label_indicator.draw(renderer, *args, **kwargs)
            self.label_text.draw(renderer, *args, **kwargs)
        super(Frame, self).draw(renderer, *args, **kwargs)


    def add_frame(self, axis):
        """Add the frame to a 3D axis."""
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            axis.add_line(b)
        if self.draw_label:
            axis.add_line(self.label_indicator)
            axis._add_text(self.label_text)


class Trajectory(artist.Artist):
    """A Matplotlib artist that displays a trajectory.

    Parameters
    ----------
    H : array-like, shape (n_steps, 4, 4)
        Sequence of poses represented by homogeneous matrices

    show_direction : bool, optional (default: True)
        Plot an arrow to indicate the direction of the trajectory

    n_frames : int, optional (default: 10)
        Number of frames that should be plotted to indicate the rotation

    s : float, optional (default: 1)
        Scaling of the frames that will be drawn

    Other arguments are passed onto Line3D.
    """
    def __init__(self, H, show_direction=True, n_frames=10, s=1.0, color='c', show_frame=True, **kwargs):
        super(Trajectory, self).__init__()

        self.show_direction = show_direction
        self.show_frame = show_frame

        self.trajectory = Line3D([], [], [], color=color, linewidth=3, **kwargs)
        self.key_frames = [Frame(np.eye(4), s=s, **kwargs)
                           for _ in range(n_frames)]

        if self.show_direction:
            self.direction_arrow = Arrow3D(
                [0, 0], [0, 0], [0, 0],
                mutation_scale=20, lw=1, arrowstyle="-|>", color="k")

        self.set_data(H)


    def set_data(self, H):
        """Set the trajectory data.

        Parameters
        ----------
        H : array-like, shape (n_steps, 4, 4)
            Sequence of poses represented by homogeneous matrices
        """
        positions = H[:, :3, 3]
        self.trajectory.set_data(positions[:, 0], positions[:, 1])
        self.trajectory.set_3d_properties(positions[:, 2])

        key_frames_indices = np.linspace(
            0, len(H) - 1, len(self.key_frames), dtype=np.int)
        # key_frames_indices = np.array([len(H) - 1], dtype=np.int)
        for i, key_frame_idx in enumerate(key_frames_indices):
            self.key_frames[i].set_data(H[key_frame_idx])

        if self.show_direction:
            start = 0.8 * positions[0] + 0.2 * positions[-1]
            end = 0.2 * positions[0] + 0.8 * positions[-1]
            self.direction_arrow.set_data(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]])


    @artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        """Draw the artist."""
        self.trajectory.draw(renderer, *args, **kwargs)
        for key_frame in self.key_frames:
            if self.show_frame:
                key_frame.draw(renderer, *args, **kwargs)
        if self.show_direction:
            self.direction_arrow.draw(renderer)
        super(Trajectory, self).draw(renderer, *args, **kwargs)


    def add_trajectory(self, axis):
        """Add the trajectory to a 3D axis."""
        axis.add_line(self.trajectory)
        for key_frame in self.key_frames:
            key_frame.add_frame(axis)
        if self.show_direction:
            axis.add_artist(self.direction_arrow)



class Arrow3D(FancyArrowPatch):  # http://stackoverflow.com/a/11156353/915743
    """A Matplotlib patch that represents an arrow in 3D."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super(Arrow3D, self).__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs


    def set_data(self, xs, ys, zs):
        """Set the arrow data.

        Parameters
        ----------
        xs : iterable
            List of x positions

        ys : iterable
            List of y positions

        zs : iterable
            List of z positions
        """
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the patch."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super(Arrow3D, self).draw(renderer)


def make_3d_axis(ax_s, pos=111):
    """Generate new 3D axis.

    Parameters
    ----------
    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    pos : int, optional (default: 111)
        Position indicator (nrows, ncols, plot_number)

    Returns
    -------
    ax : Matplotlib 3d axis
        New axis
    """
    ax = plt.subplot(pos, projection="3d", aspect="equal")
    plt.setp(ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), zlim=(-ax_s, ax_s),
             xlabel="X", ylabel="Y", zlabel="Z")
    return ax

def check_matrix(R):
    """Input validation of a rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    Returns
    -------
    R : array, shape (3, 3)
        Validated rotation matrix
    """
    R = np.asarray(R, dtype=np.float)
    if R.ndim != 2 or R.shape[0] != 3 or R.shape[1] != 3:
        raise ValueError("Expected rotation matrix with shape (3, 3), got "
                         "array-like object with shape %s" % (R.shape,))
    RRT = np.dot(R, R.T)
    if not np.allclose(RRT, np.eye(3)):
        raise ValueError("Expected rotation matrix, but it failed the test "
                         "for inversion by transposition. np.dot(R, R.T) "
                         "gives %r" % RRT)
    return R

def check_quaternion(q, unit=True):
    """Input validation of quaternion representation.

    Parameters
    ----------
    q : array-like, shape (4,)
        Quaternion to represent rotation: (w, x, y, z)

    unit : bool, optional (default: True)
        Normalize the quaternion so that it is a unit quaternion

    Returns
    -------
    q : array-like, shape (4,)
        Validated quaternion to represent rotation: (w, x, y, z)
    """
    q = np.asarray(q, dtype=np.float)
    if q.ndim != 1 or q.shape[0] != 4:
        raise ValueError("Expected quaternion with shape (4,), got "
                         "array-like object with shape %s" % (q.shape,))
    if unit:
        return norm_vector(q)
    else:
        return q

def norm_vector(v):
    """Normalize vector.

    Parameters
    ----------
    v : array-like, shape (n,)
        nd vector

    Returns
    -------
    u : array-like, shape (n,)
        nd unit vector with norm 1 or the zero vector
    """
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    else:
        return np.asarray(v) / norm
    
def matrix_from_quaternion(q):
    """Compute rotation matrix from quaternion.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    q = check_quaternion(q)
    uq = norm_vector(q)
    w, x, y, z = uq
    x2 = 2.0 * x * x
    y2 = 2.0 * y * y
    z2 = 2.0 * z * z
    xy = 2.0 * x * y
    xz = 2.0 * x * z
    yz = 2.0 * y * z
    xw = 2.0 * x * w
    yw = 2.0 * y * w
    zw = 2.0 * z * w

    R = np.array([[1.0 - y2 - z2, xy - zw, xz + yw],
                  [xy + zw, 1.0 - x2 - z2, yz - xw],
                  [xz - yw, yz + xw, 1.0 - x2 - y2]])
    return R
    
    
def matrices_from_pos_quat(P):
    """Get sequence of homogeneous matrices from positions and quaternions.

    Parameters
    ----------
    P : array-like, shape (n_steps, 7)
        Sequence of poses represented by positions and quaternions in the
        order (x, y, z, w, vx, vy, vz) for each step

    Returns
    -------
    H : array-like, shape (n_steps, 4, 4)
        Sequence of poses represented by homogeneous matrices
    """
    n_steps = len(P)
    H = np.empty((n_steps, 4, 4))
    H[:, :3, 3] = P[:, :3]
    H[:, 3, :3] = 0.0
    H[:, 3, 3] = 1.0
    for t in range(n_steps):
        H[t, :3, :3] = matrix_from_quaternion(P[t, 3:])
    return H

def find_ax_limit(P):
    xmin, xmax, ymin, ymax, zmin, zmax = P[:,0].min(), P[:,0].max(), P[:,1].min(), P[:,1].max(), P[:,2].min(), P[:,2].max()
    dd = max(max(xmax-xmin, ymax-ymin),(zmax-zmin))/2.0
    dd = max(dd, 1.0)
    xc, yc, zc = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2

    return (xc-dd, xc+dd), (yc-dd, yc+dd), (zc-dd, zc+dd) # (zmin, zmax) #

def poselist2vis(poselist):
    poselist_vis = np.zeros((len(poselist),7))
    for k,pp in enumerate(poselist):
        ppp = np.zeros(7)
        ppp[:3] = pp[:3]
        ppp[3] = pp[-1]
        ppp[4:] = pp[3:6]
        poselist_vis[k,:] = ppp
        
    return np.array(poselist_vis)
   
def plot_trajectory(P, ax, color, **kwargs):
    """
    Plot pose trajectory.
    """
    xlim, ylim, zlim = find_ax_limit(P)
    plt.setp(ax, xlim=xlim, ylim=ylim, zlim=zlim,
                 xlabel="X", ylabel="Y", zlabel="Z")

    H = matrices_from_pos_quat(P)
    trajectory = Trajectory(H, show_direction=True, n_frames=min(len(P), 100), s=(xlim[1]-xlim[0])/50, 
                                    color=color,  **kwargs)
    trajectory.add_trajectory(ax)
    
    return ax

def plot_trajectories(Ps, ax, labels=None, colors='kcmy', **kwargs):
    xlim, ylim, zlim = find_ax_limit(np.concatenate(Ps, axis=0))
    plt.setp(ax, xlim=xlim, ylim=ylim, zlim=zlim,
                 xlabel="X", ylabel="Y", zlabel="Z")
    for k, P in enumerate(Ps):
        H = matrices_from_pos_quat(P)
        if k == 0: 
            trajectory = Trajectory(H, show_direction=False, n_frames=0, s=(xlim[1]-xlim[0])/30, 
                                    color=colors[k%len(colors)],  linestyle='dashed')
        else:
            trajectory = Trajectory(H, show_direction=False, n_frames=0, s=(xlim[1]-xlim[0])/30, 
                                    color=colors[k%len(colors)],  **kwargs)
        trajectory.add_trajectory(ax)
        # plot a pint at the end of the trajectory
        ax.scatter(P[-1][0], P[-1][1], P[-1][2], marker='o', s=50, color=colors[k%len(colors)])

    if labels is not None:
        plt.legend(labels)
    return ax

def rotate_ax(step, ax):
    ax.view_init(30, step * 2) 

def plot_gt_est_trajectories(trajs, showfig=False, savefigname=None, saveaniname=None, labels=['GT Traj','Estimated Traj'], **kwargs):
    plt.close('all')
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111, projection="3d", aspect="equal")

    trajlen = len(trajs[0])
    trajs_vis = []
    for ttt in trajs:
        trajs_vis.append(poselist2vis(ttt))
    ax = plot_trajectories(trajs_vis, ax=ax, labels=labels, **kwargs)

    if savefigname is not None:
        plt.savefig(savefigname)

    if saveaniname is not None:
        import matplotlib.animation as animation 
        anim = animation.FuncAnimation(fig, rotate_ax, 180, fargs=(ax, ), interval=100, blit=False)
        # write animation to mp4 file
        # print('  ==> Save mp4 file {}'.format(saveaniname))
        # Writer = animation.FFMpegWriter(fps=20, extra_args=['-vcodec', 'libx264'])
        # anim.save(saveaniname, writer=Writer)

        # write animation to gif file
        print('  ==> Save gif file {}'.format(saveaniname))
        anim.save(saveaniname, writer='imagemagick', fps=20)


    if showfig: # bug when run together with animation
        for angle in range(0, 360, 10):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(0.001)     

def update_trajectory(step, ax, trajs_vis, labels, traj_step = 20):
    ax.clear()
    k = min(len(trajs_vis[0]), (step+1) * traj_step)
    trajs_vis_clip = [ttt[:k] for ttt in trajs_vis]
    ax = plot_trajectories(trajs_vis_clip, ax=ax, labels=labels)

def animate_gt_est_trajectories(trajs, showani=False, saveaniname=None, trajstep=7, labels=['GT Traj','Estimated Traj']):
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111, projection="3d", aspect="equal")

    trajlen = len(trajs[0])
    trajs_vis = []
    for ttt in trajs:
        trajs_vis.append(poselist2vis(ttt))
    # gt_vis = poselist2vis(gt)
    # est_vis = poselist2vis(est)

    if saveaniname is not None:
        import matplotlib.animation as animation 
        anim = animation.FuncAnimation(
            fig, update_trajectory, int(trajlen/trajstep) + 1, fargs=(ax, trajs_vis, labels, trajstep), interval=100, blit=False)

        print('  ==> Save mp4 file {}'.format(saveaniname))
        Writer = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
        anim.save(saveaniname, writer=Writer)
        # write animation to gif file
        # print('  ==> Save gif file {}'.format(saveaniname))
        # anim.save(saveaniname, writer='imagemagick', fps=20)

    if showani:
        for k in range(1,trajlen,trajstep):
            ax.clear()
            trajs_vis_clip = [ttt[:k] for ttt in trajs_vis]
            ax = plot_trajectories(trajs_vis_clip, ax=ax, labels=labels)
            plt.draw()
            plt.pause(0.01)
    

if __name__ == '__main__':
    # gtposes = np.loadtxt('data/pose_gt.txt')
    # estposes = np.loadtxt('data/pose.txt')

    gtposes = np.loadtxt('/home/wenshan/tmp/testing_traj/orbmono/sHard/carwelding_P003/pose_left.txt')
    estposes = np.loadtxt('/home/wenshan/tmp/testing_traj/orbmono/sHard/carwelding_P003/2_est.txt')

    gtposes = np.loadtxt('/home/wenshan/tmp/testing_traj/orbmono/mHard/endofworld_P004/pose_left.txt')
    estposes = np.loadtxt('/home/wenshan/tmp/testing_traj/orbmono/mHard/endofworld_P004/5_est.txt')
    # plot_gt_est_trajectories(gtposes, estposes, showfig=True)

    from trajectory_transform import trajectory_transform, rescale

    gt_traj_trans, est_traj_trans = trajectory_transform(gtposes, estposes)
    est_traj_trans, s = rescale(gt_traj_trans, est_traj_trans)

    # animate_gt_est_trajectories([gt_traj_trans, est_traj_trans], saveaniname='abandonedfactory_P003_ani.gif')
    plot_gt_est_trajectories([gt_traj_trans, est_traj_trans], showfig=True, savefigname='abandonedfactory_P003.jpg', saveaniname='myAnimation.gif')
