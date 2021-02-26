import _init_paths
import argparse
import os
import random
import time
import numpy as np
import yaml
import torch
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.logger import setup_logger
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ycb', help='dataset name')
parser.add_argument('--dataset_root', type=str, default='datasets/ycb/data', help='dataset root dir')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_refine', default=0.00003, help='learning rate of refinement')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=200, help='max number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--experiment_name', type=str, default='test', help='which experiment')
opt = parser.parse_args()

def main():
    if opt.dataset == 'ycb':
        opt.num_obj = 21
        opt.sym_list = [12, 15, 18, 19, 20]
        opt.num_points = 1000
        writer = SummaryWriter('experiments/runs/ycb/{0}'.format(opt.experiment_name))
        opt.outf = 'trained_models/ycb/{0}'.format(opt.experiment_name)
        opt.log_dir = 'experiments/logs/ycb/{0}'.format(opt.experiment_name)
        opt.repeat_num = 1
        if not os.path.exists(opt.outf):
            os.mkdir(opt.outf)
        if not os.path.exists(opt.log_dir):
            os.mkdir(opt.log_dir)
    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(num_points = opt.num_points, num_vote = 9, num_obj = opt.num_obj)
    estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_obj)
    refiner.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.lr = opt.lr_refine
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}'.format(len(dataset), len(test_dataset), opt.num_points))

    criterion = Loss(opt.num_points, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points, opt.sym_list)
    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()
    train_scalar = 0

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_loss_avg = 0.0
        train_loss = 0.0
        train_dis_avg = 0.0
        train_dis = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()
        for rep in range(opt.repeat_num):
            for i, data in enumerate(dataloader, 0):
                points, choose, img, target, model_points, model_kp, vertex_gt, idx, target_r, target_t = data
                points, choose, img, target, model_points, model_kp, vertex_gt, idx, target_r, target_t = points.cuda(), choose.cuda(), img.cuda(), target.cuda(), model_points.cuda(), model_kp.cuda(), vertex_gt.cuda(), idx.cuda(), target_r.cuda(), target_t.cuda()
                vertex_pred, c_pred, emb = estimator(img, points, choose, idx)
                vertex_loss, pose_loss, dis, new_points, new_target = criterion(vertex_pred, vertex_gt, c_pred, points, target, model_points, model_kp, idx, target_r, target_t)
                loss = 10 * vertex_loss + pose_loss
                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_points, new_target, model_points, idx)
                        dis.backward()
                else:
                    loss.backward()
                train_loss_avg += loss.item()
                train_loss += loss.item()
                train_dis_avg += dis.item()
                train_dis += dis.item()
                train_count += 1
                train_scalar += 1

                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_loss:{4} Avg_diss:{5}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_loss_avg / opt.batch_size, train_dis_avg / opt.batch_size))
                    writer.add_scalar('ycb training loss', train_loss_avg / opt.batch_size, train_scalar)
                    writer.add_scalar('ycb training dis', train_dis_avg / opt.batch_size, train_scalar)
                    optimizer.step()
                    optimizer.zero_grad()
                    train_loss_avg = 0
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))
        train_loss = train_loss / train_count
        train_dis = train_dis / train_count
        logger.info('Train time {0} Epoch {1} TRAIN FINISH Avg loss: {2} Avg dis: {3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, train_loss, train_dis))

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_loss = 0.0
        test_vertex_loss = 0.0
        test_pose_loss = 0.0
        test_dis = 0.0
        test_count = 0
        success_count = 0
        estimator.eval()
        refiner.eval()
        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, model_kp, vertex_gt, idx, target_r, target_t = data
            points, choose, img, target, model_points, model_kp, vertex_gt, idx, target_r, target_t = points.cuda(), choose.cuda(), img.cuda(), target.cuda(), model_points.cuda(), model_kp.cuda(), vertex_gt.cuda(), idx.cuda(), target_r.cuda(), target_t.cuda()
            vertex_pred, c_pred, emb = estimator(img, points, choose, idx)
            vertex_loss, pose_loss, dis, new_points, new_target = criterion(vertex_pred, vertex_gt, c_pred, points, target, model_points, model_kp, idx, target_r, target_t)
            loss = 10 * vertex_loss + pose_loss
            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_points, new_target, model_points, idx)
            test_loss += loss.item()
            test_vertex_loss += vertex_loss.item()
            test_pose_loss += pose_loss.item()
            test_dis += dis.item()
            logger.info('Test time {0} Test Frame No.{1} loss:{2} dis:{3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss, dis))
            test_count += 1
            if dis.item() < 0.02:
                success_count += 1

        test_loss = test_loss / test_count
        test_vertex_loss = test_vertex_loss / test_count
        test_pose_loss = test_pose_loss / test_count
        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg loss: {2} Avg dis: {3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_loss, test_dis))
        logger.info('Success rate: {}'.format(float(success_count) / test_count))
        writer.add_scalar('ycb test loss', test_loss, epoch)
        writer.add_scalar('ycb test vertex loss', test_vertex_loss, epoch)
        writer.add_scalar('ycb test pose loss', test_pose_loss, epoch)
        writer.add_scalar('ycb test dis', test_dis, epoch)
        writer.add_scalar('ycb success rate', float(success_count) / test_count, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        if test_dis <= best_test:
            best_test = test_dis
        if opt.refine_start:
            torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
        else:
            torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
        print(epoch, '>>>>>>>>----------MODEL SAVED---------<<<<<<<<')

        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.lr = opt.lr_refine
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
            print('>>>>>>>>----------Refine started---------<<<<<<<<')

    writer.close()

if __name__ == '__main__':
    main()
