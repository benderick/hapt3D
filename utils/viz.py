import open3d as o3d
import matplotlib
import distinctipy
import numpy as np

'''
SEMANTIC:
    0: void - black
    1: ground - tan
    2: plant - seagreen
    3: fruit - orangered
    4: trunk - saddlebrown
    5: pole - slategray

'''

def labels2colors(points, labels, attribute, show=True):
    
    pcd = o3d.geometry.PointCloud()

    if 'ins' in attribute:
        uni_lables, inv, count = np.unique(labels,return_counts=True, return_inverse=True)
        N = len(uni_lables)
        colors_list = distinctipy.get_colors(N, exclude_colors=[(1*c,1*c,1*c) for c in np.arange(0,1.01,0.01)])
        colors_list[np.argmax(count)] = (0.7, 0.7, 0.7)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors_list, len(colors_list))
        labels_norm = labels / len(colors_list)
        colors = cmap(labels_norm)[:, :-1]
    elif 'sem' in attribute:
        colors_list =  ["black","tan","seagreen","orangered","saddlebrown","slategray"]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors_list, len(colors_list))
        labels_norm = labels / len(colors_list)
        colors = cmap(labels_norm)[:, :-1]
    else:
        raise NotImplementedError        

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if show:
        o3d.visualization.draw_geometries([pcd],window_name=attribute)
    return pcd
