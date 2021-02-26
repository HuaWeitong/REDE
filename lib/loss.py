import torch
from torch.nn.modules.loss import _Loss
from lib.KNN_CUDA.knn_cuda import KNN
from torch_batch_svd import svd
import torchgeometry as tgm

def smooth_l1_loss(vertex_pred, vertex_targets, sigma=1.0, normalize=True):
    b,ver_dim,num_points=vertex_pred.shape
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = vertex_diff
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
              + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (ver_dim * num_points)

    return in_loss


def batch_least_square(A, B, w):

    assert A.shape == B.shape
    num = A.shape[0]
    centroid_A = torch.mean(A, dim=1)
    centroid_B = torch.mean(B, dim=1)
    AA = A - centroid_A.unsqueeze(1)
    BB = B - centroid_B.unsqueeze(1)

    H = torch.bmm(torch.transpose(AA, 2, 1), BB)
    U, S, Vt = svd(H)

    R = torch.bmm(Vt, U.permute(0, 2, 1))
    i = torch.det(R) < 0
    tmp = torch.ones([num, 3, 3], dtype=torch.float32).cuda()
    tmp[i, :, 2] = -1
    Vt = Vt * tmp

    R = torch.bmm(Vt, U.permute(0, 2, 1))
    t = centroid_B - torch.bmm(R, centroid_A.unsqueeze(2)).squeeze()
    return R, t


def calculate_error(r, t, model_points, scene_points):
    pred = torch.bmm(model_points[0].expand(r.shape[0], model_points.shape[1], 3), r.permute(0, 2, 1)) + t.expand(model_points.shape[1],r.shape[0],3).permute(1,0,2)
    knn = KNN(k=1, transpose_mode=True)
    target = scene_points[0].expand(r.shape[0], scene_points.shape[1], 3)
    dist, inds = knn(pred, target)
    dis = torch.mean(dist.squeeze(), 1)
    return dis


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, vertex_pred, vertex_gt, c_pred, points, target, model_points, model_kp, idx, target_r, target_t):
        vertex_loss = smooth_l1_loss(vertex_pred.view(1, self.num_pt_mesh, -1), vertex_gt.view(1, self.num_pt_mesh, -1))

        kp_set = vertex_pred + points.repeat(1, 1, 9).view(1, points.shape[1], 9, 3)
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
                            (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p,1), \
                            (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1), \
                            (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                            (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
        pred_r = pred_r.squeeze()
        pred_t = torch.sum(w * all_t, 0)

        target_r = target_r.squeeze()
        target_t = target_t.squeeze()
        pose_loss = torch.norm(pred_t - target_t) + 0.01 * torch.norm(torch.mm(pred_r, torch.transpose(target_r, 1, 0)) - torch.eye(3).cuda())

        pred = torch.mm(model_points[0], torch.transpose(pred_r, 1, 0)) + pred_t
        knn = KNN(k=1, transpose_mode=True)
        if idx in self.sym_list:
            dist, inds = knn(pred.unsqueeze(0), target.unsqueeze(0))
            dis = torch.mean(dist.squeeze())
        else:
            dis = torch.mean(torch.norm(pred - target[0], dim=1), dim=0)

        ori_r = torch.unsqueeze(pred_r, 0).cuda()
        ori_t = torch.unsqueeze(pred_t, 0).cuda()
        ori_t = ori_t.repeat(self.num_pt_mesh, 1).contiguous().view(1, self.num_pt_mesh, 3)
        new_points = torch.bmm((points - ori_t), ori_r).contiguous()

        new_target = torch.bmm((target - ori_t), ori_r).contiguous()

        del knn

        return vertex_loss, pose_loss, dis, new_points.detach(), new_target.detach()
