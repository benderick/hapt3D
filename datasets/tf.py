import numpy as np
import scipy
import open3d as o3d
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def get_geom_aug(config):

    # 1. Scale
    I = np.eye(4)
    scale = np.random.uniform(config['transform']['scale']['min_factor'],
                            config['transform']['scale']['max_factor'], size=(1,4)) 
    # don't scale the homogeneous coord !
    scale[0,-1] = 1
    T = scale*I

    # 2. Rotation around Z
    angle = np.random.uniform(-config['transform']['rotation']['max_z']* np.pi/180.0,
                            +config['transform']['rotation']['max_z']* np.pi/180.0)
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.asarray([[0,0,angle]]).T)
    T_R = np.eye(4)
    T_R[0:3,0:3] = R
    T = T_R @ T

    # 3. Rotation around Y
    angle = np.random.uniform(-config['transform']['rotation']['max_y']* np.pi/180.0,
                            +config['transform']['rotation']['max_y']* np.pi/180.0)
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.asarray([[0,angle,0]]).T)
    T_R = np.eye(4)
    T_R[0:3,0:3] = R
    T = T_R @ T

    # 4. Rotation around X
    angle = np.random.uniform(-config['transform']['rotation']['max_x']* np.pi/180.0,
                            +config['transform']['rotation']['max_x']* np.pi/180.0)
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.asarray([[angle,0,0]]).T)
    T_R = np.eye(4)
    T_R[0:3,0:3] = R
    T = T_R @ T

    # 5. Shear 
    shear = np.random.uniform(-config['transform']['shear']['max_factor'], +config['transform']['shear']['max_factor'], size=(3,)) 
    T_shear = np.eye(4)
    T_shear[0,1] = shear[0] # xy
    T_shear[0,2] = shear[1] # xz
    T_shear[1,2] = shear[2] # yz
    T = T_shear @ T

    return T

# Color jittering  
def color_jitter(config, colors):
    # a x + b
    # a > 1 = more contrast
    # 0 < a < 1 = less contrast
    # b = brightness
    contrast = np.random.uniform(config['transform']['color']['contrast'][0], config['transform']['color']['contrast'][1]) 
    brightness = np.random.uniform(-config['transform']['color']['brightness'], +config['transform']['color']['brightness']) 
    colors = contrast * colors + brightness
    try:
        hsv = rgb_to_hsv(colors.numpy())
    except IndexError:
        return colors
    hue = np.random.uniform(-config['transform']['color']['hue'], config['transform']['color']['hue']) 
    saturation = np.random.uniform(-config['transform']['color']['saturation'], config['transform']['color']['saturation']) 
    hsv[:, 0] += hue
    hsv[:, 1] += saturation
    colors = o3d.core.Tensor(hsv_to_rgb(hsv))
    return colors

# Elastic distortion
def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.
    Thanks to chrischoy for providing the code at https://github.com/chrischoy/SpatioTemporalSegmentation.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)
    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(
            noise, blurx, mode="constant", cval=0
        )
        noise = scipy.ndimage.filters.convolve(
            noise, blury, mode="constant", cval=0
        )
        noise = scipy.ndimage.filters.convolve(
            noise, blurz, mode="constant", cval=0
        )

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud