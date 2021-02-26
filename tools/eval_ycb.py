import _init_paths
import argparse
import random
import time
import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import torch.utils.data
import torchvision.transforms as transforms
import torchgeometry as tgm
from datasets.ycb.dataset import get_bbox
from lib.network import PoseNet, PoseRefineNet
from lib.KNN_CUDA.knn_cuda import KNN
from lib.loss import calculate_error, batch_least_square
from lib.transformations import quaternion_matrix, quaternion_from_matrix
from tools.utils import icp
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = 'datasets/ycb/data', help='dataset root dir')
opt = parser.parse_args()

trained_model = 'trained_checkpoints/ycb/pose_model_25_0.012096800266755742.pth'
refine_model = 'trained_checkpoints/ycb/pose_refine_model_121_0.008576695929338575.pth'
output_result_dir = 'experiments/eval_result/ycb'
fw = open('{0}/res.txt'.format(output_result_dir), 'w')

obj_num = 21
sym_list = [12, 15, 18, 19, 20]
num_points = 1000
num_vote = 9
iteration = 2
success_count = [0 for i in range(obj_num)]
success_count_adds = [0 for i in range(obj_num)]
success_count_icp = [0 for i in range(obj_num)]
success_count_adds_icp = [0 for i in range(obj_num)]
num_count = [0 for i in range(obj_num)]
add = [[] for i in range(obj_num)]
adds = [[] for i in range(obj_num)]
add_icp = [[] for i in range(obj_num)]
adds_icp = [[] for i in range(obj_num)]

knn = KNN(k=1, transpose_mode=True)
estimator = PoseNet(num_points=num_points, num_vote=num_vote, num_obj = obj_num)
estimator.cuda()
estimator.load_state_dict(torch.load(trained_model))
estimator.eval()
refiner = PoseRefineNet(num_points=num_points, num_obj = obj_num)
refiner.cuda()
refiner.load_state_dict(torch.load(refine_model))
refiner.eval()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
img_width = 480
img_length = 640

dataset_config_dir = 'datasets/ycb/dataset_config'
testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
kp = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    kp[class_id] = np.loadtxt('{0}/keypoints/fps_{1}.txt'.format(opt.dataset_root, '%02d' % class_id))
    class_id += 1

for now in range(0, len(testlist)):
    img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
    depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
    meta = scio.loadmat('{0}/{1}-meta.mat'.format(opt.dataset_root, testlist[now]))
    label = np.array(Image.open('{0}/{1}-label.png'.format(opt.dataset_root, testlist[now])))
    lst = meta['cls_indexes'].flatten()

    for idx in range(len(lst)):
        itemid = lst[idx]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
        mask = mask_label * mask_depth

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img_masked = np.array(img)[:, :, :3]
        img_masked = np.transpose(img_masked, (2, 0, 1))
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        dellist = [j for j in range(0, len(cld[itemid]))]
        dellist = random.sample(dellist, len(cld[itemid]) - num_points)
        model_points = np.delete(cld[itemid], dellist, axis=0)

        ii = np.where(meta['cls_indexes'] == itemid)[0][0]
        target_r = meta['poses'][:, :, ii][:, 0:3]
        target_t = np.array([meta['poses'][:, :, ii][:, 3:4].flatten()])
        target = np.dot(model_points, target_r.T)
        target = np.add(target, target_t)

        model_kp = kp[itemid]

        points = torch.from_numpy(cloud.astype(np.float32))
        choose = torch.LongTensor(choose.astype(np.int32))
        img_masked = norm(torch.from_numpy((img_masked / 255).astype(np.float32)))
        index = torch.LongTensor([itemid - 1])
        model_points = torch.from_numpy(model_points.astype(np.float32))
        model_kp = torch.from_numpy(model_kp.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))

        points = points.unsqueeze(0).cuda()
        choose = choose.unsqueeze(0).cuda()
        img_masked = img_masked.unsqueeze(0).cuda()
        index = index.unsqueeze(0).cuda()
        model_points = model_points.unsqueeze(0).cuda()
        model_kp = model_kp.unsqueeze(0).cuda()
        target = target.unsqueeze(0).cuda()

        id = index.item()
        num_count[id] += 1
        if len(points.size()) == 2:
            print('Obj{0} No.{0} NOT Pass! Lost detection!'.format(id, now))
            fw.write('Obj{0} No.{0} NOT Pass! Lost detection!\n'.format(id, now))
            continue
        vertex_pred, c_pred, emb = estimator(img_masked, points, choose, index)
        kp_set = vertex_pred + points.repeat(1, 1, 9).view(1, num_points, 9, 3)
        confidence = c_pred / (0.00001 + torch.sum(c_pred, 1))
        points_pred = torch.sum(confidence * kp_set, 1)

        all_index = torch.combinations(torch.arange(9), 3)
        all_r, all_t = batch_least_square(model_kp.squeeze()[all_index, :], points_pred.squeeze()[all_index, :], torch.ones([all_index.shape[0], 3]).cuda())
        all_e = calculate_error(all_r, all_t, model_points, points)
        e = all_e.unsqueeze(0).unsqueeze(2)
        w = torch.softmax(1 / e, 1).squeeze().unsqueeze(1)
        all_qua = tgm.rotation_matrix_to_quaternion(torch.cat((all_r, torch.tensor([0., 0., 1.]).cuda().unsqueeze(1).repeat(all_index.shape[0], 1, 1)), dim=2))
        pred_qua = torch.sum(w * all_qua, 0)
        pred_r = pred_qua.view(1, 1, -1)
        bs, num_p, _ = pred_r.size()
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        pred_r = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                            (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1), \
                            (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                            (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
        my_r = pred_r.squeeze()
        my_t = torch.sum(w * all_t, 0)
        my_r = my_r.cpu().detach().numpy()
        my_t = my_t.cpu().detach().numpy()

        for ite in range(0, iteration):
            R = torch.unsqueeze(torch.from_numpy(my_r.astype(np.float32)), 0).cuda()
            ori_t = torch.unsqueeze(torch.from_numpy(my_t.astype(np.float32)), 0).cuda()
            T = ori_t.repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = np.column_stack((my_r, my_t))
            my_mat = np.row_stack((my_mat, [0, 0, 0, 1]))

            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = refiner(new_points, emb, index)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2

            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r = my_mat_final[:3, :3]
            my_t = my_mat_final[:3, 3]

        model_points = model_points[0].cpu().detach().numpy()
        pred = np.dot(model_points, my_r.T) + my_t
        target = target[0].cpu().detach().numpy()

        dis_add = np.mean(np.linalg.norm(pred - target, axis=1))
        add[id].append(dis_add)
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().contiguous()
        dist, inds = knn(pred.unsqueeze(0), target.unsqueeze(0))
        dis_adds = torch.mean(dist.squeeze())
        adds[id].append(dis_adds)
        if id in sym_list:
            dis = dis_adds
        else:
            dis = dis_add

        target = target.cpu().detach().numpy()
        points = points.cpu().detach().numpy()[0]
        RT = np.column_stack((my_r, my_t))
        RT = np.row_stack((RT, [0, 0, 0, 1]))
        RT_icp, distances, iterations = icp.my_icp(model_points, points, init_pose=RT)
        my_r = RT_icp[:3, :3]
        my_t = RT_icp[:3, 3]
        pred = np.dot(model_points, my_r.T) + my_t

        dis_add_icp = np.mean(np.linalg.norm(pred - target, axis=1))
        add_icp[id].append(dis_add_icp)
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().contiguous()
        dist, inds = knn(pred.unsqueeze(0), target.unsqueeze(0))
        dis_adds_icp = torch.mean(dist.squeeze())
        adds_icp[id].append(dis_adds_icp)
        if id in sym_list:
            dis_icp = dis_adds_icp
        else:
            dis_icp = dis_add_icp

        if dis < 0.02:
            success_count[id] += 1
            print('No.{0} Obj{1} Pass! dis: {2} adds: {3} icp: {4} adds_icp: {5}'.format(now, id, dis, dis_adds, dis_icp, dis_adds_icp))
            fw.write('No.{0} Obj{1} Pass! dis: {2} adds: {3} icp: {4} adds_icp: {5}\n'.format(now, id, dis, dis_adds, dis_icp, dis_adds_icp))
        else:
            print('No.{0} Obj{1} NOT Pass! dis: {2} adds: {3} icp: {4} adds_icp: {5}'.format(now, id, dis, dis_adds, dis_icp, dis_adds_icp))
            fw.write('No.{0} Obj{1} NOT Pass! dis: {2} adds: {3} icp: {4} adds_icp: {5}\n'.format(now, id, dis, dis_adds, dis_icp, dis_adds_icp))
        if dis_adds < 0.02:
            success_count_adds[id] += 1
        if dis_icp < 0.02:
            success_count_icp[id] += 1
        if dis_adds_icp < 0.02:
            success_count_adds_icp[id] += 1

print("########## adds ##########")
fw.write("########## adds ##########\n")
for i in range(obj_num):
    print('Object {0} success rate: {1}'.format(i, float(success_count_adds[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(i, float(success_count_adds[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count_adds)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count_adds)) / sum(num_count)))

print("########## add(s) ##########")
fw.write("########## add(s) ##########\n")
for i in range(obj_num):
    print('Object {0} success rate: {1}'.format(i, float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(i, float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))

print("########## adds_icp ##########")
fw.write("########## adds_icp ##########\n")
for i in range(obj_num):
    print('Object {0} success rate: {1}'.format(i, float(success_count_adds_icp[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(i, float(success_count_adds_icp[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count_adds_icp)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count_adds_icp)) / sum(num_count)))

print("########## add(s)_icp ##########")
fw.write("########## add(s)_icp ##########\n")
for i in range(obj_num):
    print('Object {0} success rate: {1}'.format(i, float(success_count_icp[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(i, float(success_count_icp[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count_icp)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count_icp)) / sum(num_count)))

def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap

def cal_auc(add_dis, max_dis=0.1):
    D = np.array(add_dis)
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(add_dis)
    acc = np.cumsum(np.ones((1, n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    return aps * 100

add_auc = [0 for i in range(obj_num)]
adds_auc = [0 for i in range(obj_num)]
add_s_auc = [0 for i in range(obj_num)]
add_auc_icp = [0 for i in range(obj_num)]
adds_auc_icp = [0 for i in range(obj_num)]
add_s_auc_icp = [0 for i in range(obj_num)]

for i in range(obj_num):
    add_auc[i] = cal_auc(add[i])
    adds_auc[i] = cal_auc(adds[i])
    if i in sym_list:
        add_s_auc[i] = adds_auc[i]
    else:
        add_s_auc[i] = add_auc[i]
    add_auc_icp[i] = cal_auc(add_icp[i])
    adds_auc_icp[i] = cal_auc(adds_icp[i])
    if i in sym_list:
        add_s_auc_icp[i] = adds_auc_icp[i]
    else:
        add_s_auc_icp[i] = add_auc_icp[i]

print("########## adds ##########")
fw.write("########## adds ##########\n")
for i in range(obj_num):
    print('Object {0}: {1}'.format(i, adds_auc[i]))
    fw.write('Object {0}: {1}\n'.format(i, adds_auc[i]))
print('ALL AUC: {0}'.format(sum(adds_auc) / obj_num))
fw.write('ALL AUC: {0}\n'.format(sum(adds_auc) / obj_num))

print("########## add(s) ##########")
fw.write("########## add(s) ##########\n")
for i in range(obj_num):
    print('Object {0}: {1}'.format(i, add_s_auc[i]))
    fw.write('Object {0}: {1}\n'.format(i, add_s_auc[i]))
print('ALL AUC: {0}'.format(sum(add_s_auc) / obj_num))
fw.write('ALL AUC: {0}\n'.format(sum(add_s_auc) / obj_num))

print("########## adds_icp ##########")
fw.write("########## adds_icp ##########\n")
for i in range(obj_num):
    print('Object {0}: {1}'.format(i, adds_auc_icp[i]))
    fw.write('Object {0}: {1}\n'.format(i, adds_auc_icp[i]))
print('ALL AUC: {0}'.format(sum(adds_auc_icp) / obj_num))
fw.write('ALL AUC: {0}\n'.format(sum(adds_auc_icp) / obj_num))

print("########## add(s)_icp ##########")
fw.write("########## add(s)_icp ##########\n")
for i in range(obj_num):
    print('Object {0}: {1}'.format(i, add_s_auc_icp[i]))
    fw.write('Object {0}: {1}\n'.format(i, add_s_auc_icp[i]))
print('ALL AUC: {0}'.format(sum(add_s_auc_icp) / obj_num))
fw.write('ALL AUC: {0}\n'.format(sum(add_s_auc_icp) / obj_num))
fw.close()
