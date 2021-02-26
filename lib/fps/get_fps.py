import os
import numpy as np
import fps_utils

def read_ply_points(ply_path):
    f = open(ply_path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

def read_xyz_points(xyz_path):
    input_file = open(xyz_path)
    cld = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1].split(' ')
        cld.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    points = np.array(cld)
    return points

def sample_fps_points(save_dir, model_root, i):
    if model_root.find('ply') != -1:
        model_points = read_ply_points(model_root) / 1000
    elif model_root.find('xyz') != -1:
        model_points = read_xyz_points(model_root)
    farthest = fps_utils.farthest_point_sampling(model_points, 8, True)
    # np.savetxt('{}/farthest_{}.txt'.format(dataset, '%02d' % i), farthest)
    min_x, max_x = np.min(model_points[:, 0]), np.max(model_points[:, 0])
    min_y, max_y = np.min(model_points[:, 1]), np.max(model_points[:, 1])
    min_z, max_z = np.min(model_points[:, 2]), np.max(model_points[:, 2])
    corner_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    # np.savetxt('{}/corner_{}.txt'.format(dataset, '%02d' % i), corner_3d)
    # np.savetxt('{}/center_{}.txt'.format(dataset, '%02d' % i), center_3d.reshape(1,3))
    fps = np.append(farthest, center_3d.reshape(1,3), axis=0)
    np.savetxt('{}/fps_{}.txt'.format(save_dir, '%02d' % i), fps)

def main():
    dataset = 'ycb'
    save_dir = 'datasets/{}/data/keypoints'.format(dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if dataset == 'linemod':
        model_dir = '/data1/weitong/dataset/Linemod_preprocessed/models'
        obj_list = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        for i in obj_list:
            model = '{}/obj_{}.ply'.format(model_dir, '%02d' % i)
            sample_fps_points(save_dir, model, i)
    elif dataset == 'ycb':
        model_dir = '/data1/weitong/dataset/YCB_Video_Dataset/models'
        obj_list = []
        class_file = open('datasets/ycb/dataset_config/classes.txt')
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            class_input = class_input[:-1]
            obj_list.append(class_input)
        for i, obj in enumerate(obj_list):
            model = '{}/{}/points.xyz'.format(model_dir, obj)
            sample_fps_points(save_dir, model, i+1)

if __name__ == '__main__':
    main()
