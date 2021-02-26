import random
import numpy as np
import numpy.ma as ma
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms


def loaddata(img, depth, label, pose, model_file, kp):

    num_points = 500
    cam_cx = 325.26110
    cam_cy = 242.04899
    cam_fx = 572.41140
    cam_fy = 573.57043
    
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    if len(label.shape) == 2:
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
    else:
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
    mask = mask_label * mask_depth

    img = np.array(img)[:, :, :3]
    img = np.transpose(img, (2, 0, 1))
    img_masked = img
    rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
    img_masked = img_masked[:, rmin:rmax, cmin:cmax]

    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) == 0:
        cc = torch.LongTensor([0])
        return(cc, cc, cc, cc, cc, cc, cc)

    if len(choose) > num_points:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    choose = np.array([choose])
    cam_scale = 1.0
    pt2 = depth_masked * cam_scale
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1) / 1000.0

    model_points = ply_vtx(model_file) / 1000.0
    dellist = [j for j in range(0, len(model_points))]
    dellist = random.sample(dellist, len(model_points) - num_points)
    model_points = np.delete(model_points, dellist, axis=0)

    target_r = pose[:3, :3]
    target_t = pose[:3, 3]
    target = np.dot(model_points, target_r.T)
    target = np.add(target, target_t)

    model_kp = kp
    scene_kp = np.add(np.dot(model_kp, target_r.T), target_t)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0), \
           torch.LongTensor(choose.astype(np.int32)).unsqueeze(0), \
           norm(torch.from_numpy((img_masked/255).astype(np.float32))).unsqueeze(0), \
           torch.from_numpy(target.astype(np.float32)).unsqueeze(0), \
           torch.from_numpy(model_points.astype(np.float32)).unsqueeze(0), \
           torch.from_numpy(model_kp.astype(np.float32)), \
           torch.from_numpy(scene_kp.astype(np.float32))

def loaddata_occ(img, depth, label, pose, model_file, kp):

    num_points = 500
    cam_cx = 325.26110
    cam_cy = 242.04899
    cam_fx = 572.41140
    cam_fy = 573.57043
    
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(1)))
    mask = mask_label * mask_depth

    img = np.array(img)[:, :, :3]
    img = np.transpose(img, (2, 0, 1))
    img_masked = img
    rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
    img_masked = img_masked[:, rmin:rmax, cmin:cmax]
    
    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) == 0:
        cc = torch.LongTensor([0])
        return(cc, cc, cc, cc, cc)

    if len(choose) > num_points:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    choose = np.array([choose])
    cam_scale = 1.0
    pt2 = depth_masked * cam_scale
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1) / 1000.0

    model_points = ply_vtx(model_file) / 1000.0
    dellist = [j for j in range(0, len(model_points))]
    dellist = random.sample(dellist, len(model_points) - num_points)
    model_points = np.delete(model_points, dellist, axis=0)

    target_r = pose[:3, :3]
    target_t = pose[:3, 3]
    target = np.dot(model_points, target_r.T)
    target = np.add(target, target_t)

    model_kp = kp
    scene_kp = np.add(np.dot(model_kp, target_r.T), target_t)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0), \
           torch.LongTensor(choose.astype(np.int32)).unsqueeze(0), \
           norm(torch.from_numpy((img_masked/255).astype(np.float32))).unsqueeze(0), \
           torch.from_numpy(target.astype(np.float32)).unsqueeze(0), \
           torch.from_numpy(model_points.astype(np.float32)).unsqueeze(0), \
           torch.from_numpy(model_kp.astype(np.float32)), \
           torch.from_numpy(scene_kp.astype(np.float32))

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
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
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