import _init_paths
import argparse
import os
import numpy as np
import yaml
import torch.utils.data
import torchgeometry as tgm
from datasets.occlusion_linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.KNN_CUDA.knn_cuda import KNN
from lib.loss import calculate_error, batch_least_square
from lib.transformations import quaternion_matrix, quaternion_from_matrix
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = 'datasets/occlusion_linemod/data', help='dataset root dir')
opt = parser.parse_args()

all_obj_list = [1,2,4,5,6,8,9,10,11,12,13,14,15]
obj_list = [1,5,6,8,9,10,11,12]
obj_num = len(obj_list)

sym_list = [10,11]
num_points = 500
num_vote = 9
iteration = 2
diameter = []
meta_file = open('{0}/models/models_info.yml'.format(opt.dataset_root), 'r')
meta = yaml.load(meta_file)
for obj in obj_list:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(diameter)
success_count = [0 for i in range(obj_num)]
num_count = [0 for i in range(obj_num)]
total_dis = [0 for i in range(obj_num)]

knn = KNN(k=1, transpose_mode=True)
obj_name_list = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
output_result_dir = 'experiments/eval_result/occ'

for id, obj_id in enumerate(obj_list):
    obj_name = obj_name_list[all_obj_list.index(obj_id)]
    fw = open('{0}/{1}.txt'.format(output_result_dir, obj_name), 'w')
    model_list = sorted(os.listdir('trained_checkpoints/linemod/{}'.format(obj_name)))
    trained_model = 'trained_checkpoints/linemod/{}/{}'.format(obj_name, model_list[0])
    refine_model = 'trained_checkpoints/linemod/{}/{}'.format(obj_name, model_list[1])

    estimator = PoseNet(num_points=num_points, num_vote=num_vote, num_obj = 1)
    estimator.cuda()
    estimator.load_state_dict(torch.load(trained_model))
    estimator.eval()
    refiner = PoseRefineNet(num_points=num_points, num_obj = 1)
    refiner.cuda()
    refiner.load_state_dict(torch.load(refine_model))
    refiner.eval()
    testdataset = PoseDataset_linemod('eval', num_points, opt.dataset_root, obj_id)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1)

    for i, data in enumerate(testdataloader, 0):
        points, choose, img, target, model_points, model_kp, vertex_gt, idx, target_r, target_t = data
        num_count[id] += 1
        if len(points.size()) == 2:
            print('Obj{0} No.{0} NOT Pass! Lost detection!'.format(obj_id, i))
            fw.write('Obj{0} No.{0} NOT Pass! Lost detection!\n'.format(obj_id, i))
            continue
        points, choose, img, target, model_points, model_kp, vertex_gt, idx, target_r, target_t = points.cuda(), choose.cuda(), img.cuda(), target.cuda(), model_points.cuda(), model_kp.cuda(), vertex_gt.cuda(), idx.cuda(), target_r.cuda(), target_t.cuda()
        vertex_pred, c_pred, emb = estimator(img, points, choose, idx)
        kp_set = vertex_pred + points.repeat(1, 1, 9).view(1, 500, 9, 3)
        confidence = c_pred / (0.00001 + torch.sum(c_pred, 1))
        points_pred = torch.sum(confidence * kp_set, 1)

        all_index = torch.combinations(torch.arange(9), 3)
        all_r, all_t = batch_least_square(model_kp.squeeze()[all_index, :], points_pred.squeeze()[all_index, :], torch.ones([all_index.shape[0], 3]).cuda())
        all_e = calculate_error(all_r, all_t, model_points, points)
        e = all_e.unsqueeze(0).unsqueeze(2)
        w = torch.softmax(1/e, 1).squeeze().unsqueeze(1)
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
            pred_r, pred_t = refiner(new_points, emb, idx)
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

        if obj_id in sym_list:
            pred = torch.from_numpy(pred.astype(np.float32)).cuda().contiguous()
            target = torch.from_numpy(target.astype(np.float32)).cuda().contiguous()
            dist, inds = knn(pred.unsqueeze(0), target.unsqueeze(0))
            dis = torch.mean(dist.squeeze())
        else:
            dis = np.mean(np.linalg.norm(pred - target, axis=1))
        total_dis[id] += dis

        if dis < diameter[id]:
            success_count[id] += 1
            print('Obj{0} No.{1} Pass! Distance: {2}'.format(obj_id, i, dis))
            fw.write('Obj{0} No.{1} Pass! Distance: {2}\n'.format(obj_id, i, dis))
        else:
            print('Obj{0} No.{1} NOT Pass! Distance: {2}'.format(obj_id, i, dis))
            fw.write('Obj{0} No.{1} NOT Pass! Distance: {2}\n'.format(obj_id, i, dis))
    fw.close()

fw = open('{0}/occ_res.txt'.format(output_result_dir), 'w')
for i, obj_id in enumerate(obj_list):
    print('Object {0} average distance: {1}'.format(obj_id, total_dis[i] / num_count[i]))
    fw.write('Object {0} average distance: {1}\n'.format(obj_id, total_dis[i] / num_count[i]))
print('ALL average distance: {0}'.format(float(sum(total_dis)) / sum(num_count)))
fw.write('ALL average distance: {0}\n'.format(float(sum(total_dis)) / sum(num_count)))
for i, obj_id in enumerate(obj_list):
    print('Object {0} success rate: {1}'.format(obj_id, float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(obj_id, float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.close()
