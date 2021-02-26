import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import numpy as np
import numpy.ma as ma
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import pickle
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, root, obj_id):
        self.obj_id = obj_id
        self.allobjlist = [1,2,4,5,6,8,9,10,11,12,13,14,15]
        self.objlist = [1,5,6,8,9,10,11,12]
        self.mode = mode
        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_pose = []
        self.root = root

        item_count = 0
        input_file = open('{0}/{1}.txt'.format(self.root, '%02d' % self.obj_id))
        while 1:
            item_count += 1
            input_line = input_file.readline()
            if self.mode == 'test' and item_count % 10 != 0:
                continue
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.list_rgb.append('{0}/rgb/color_{1}.png'.format(self.root, '%05d' % int(input_line)))
            self.list_depth.append('{0}/depth/depth_{1}.png'.format(self.root, '%05d' % int(input_line)))
            if self.mode == 'eval':
                self.list_label.append('{0}/seg/{1}/{2}.png'.format(self.root, '%02d' % self.obj_id, input_line))
            else:
                self.list_label.append('{0}/mask/{1}/{2}.png'.format(self.root, '%02d' % self.obj_id, input_line))
            self.list_pose.append('{0}/pose/{1}/pose{2}.npy'.format(self.root, '%02d' % self.obj_id, input_line))

        self.pt = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % self.obj_id))
        self.kp = np.loadtxt('{0}/keypoints/fps_{1}.txt'.format(self.root, '%02d' % self.obj_id))
        print("Object {0} buffer loaded".format(self.obj_id))

        self.length = len(self.list_rgb)
        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.num = num
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        pose = np.load(self.list_pose[index])
        cx = self.cam_cx
        cy = self.cam_cy
        fx = self.cam_fx
        fy = self.cam_fy
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(1)))
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask = mask_label * mask_depth

        img = np.array(img)[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
        img_masked = img
        rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc, cc, cc, cc, cc)
        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])
        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cx) * pt2 / fx
        pt1 = (xmap_masked - cy) * pt2 / fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1) / 1000

        model_points = self.pt / 1000
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num)
        model_points = np.delete(model_points, dellist, axis=0)

        target_r = pose[:3, :3]
        target_t = pose[:3, 3]
        target = np.dot(model_points, target_r.T)
        target = np.add(target, target_t)

        model_kp = self.kp
        scene_kp = np.add(np.dot(model_kp, target_r.T), target_t)
        vertex_gt = compute_vertex_hcoords(cloud, scene_kp)

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy((img_masked/255).astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.from_numpy(model_kp.astype(np.float32)), \
               torch.from_numpy(vertex_gt.astype(np.float32)), \
               torch.LongTensor([0]), \
               torch.from_numpy(target_r.astype(np.float32)), \
               torch.from_numpy(target_t.astype(np.float32))

    def __len__(self):
        return self.length


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def ply_vtx(path):
    f = open(path)
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

def compute_vertex_hcoords(points, hcoords):
    m = hcoords.shape[0]
    m_matrix = np.ones((1, m, 1))
    vertex = points[:, None, :] * m_matrix
    vertex = hcoords[None, :, :] - vertex
    return vertex

def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x_l = 640
    y_l = 480
    x_r = 0
    y_r = 0
    # x = 0
    # y = 0
    # w = 0
    # h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        # if tmp_w * tmp_h > w * h:
        #     x = tmp_x
        #     y = tmp_y
        #     w = tmp_w
        #     h = tmp_h
        if tmp_x < x_l:
            x_l = tmp_x
        if tmp_y < y_l:
            y_l = tmp_y
        if tmp_x + tmp_w > x_r:
            x_r = tmp_x + tmp_w
        if tmp_y + tmp_h > y_r:
            y_r = tmp_y + tmp_h
    x = x_l
    y = y_l
    w = x_r - x_l
    h = y_r - y_l
    return [x, y, w, h]

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax

def get_bbox_mask(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    if (np.where(rows)[0]).shape[0]==0 or (np.where(cols)[0]).shape[0]==0:
        return 0,0,0,0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax