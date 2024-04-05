"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate') #权重衰减10^-4
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
# 测试集样本量固定，要把测试集样本测试完
# 而batch_size就是每次测试的样本量，而2464/24=102.8次
'''
points:torch.Size([12, 1024, 6]) [B,N,D]
target:torch.Size([12]) 十二张图像的标签
'''
def test(model, loader, num_class=40):
    mean_correct = [] #用于存储每个批次中正确分类的样本数量与总样本数量的比值。
    class_acc = np.zeros((num_class, 3)) # 用于存储每个类别的准确率信息，初始化为一个num_class x 3的零矩阵。
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        '''当前测试批次'''
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
        points = points.transpose(2, 1)

        pred, _ = classifier(points) # 使用分类模型得到各个类别的预测值
        pred_choice = pred.data.max(1)[1]# 在各预测值中找出最大值的类别 
        # 计算类别准确率
        #遍历所有唯一的类别标签。
        #对于每个类别，计算预测正确的样本数量，并更新class_acc矩阵。class_acc[:, 0]存储每个类别正确预测数量，class_acc[:, 1]存储每个类别的样本数量。
        for cat in np.unique(target.cpu()):
            # classacc变量存储了对于类别cat，模型预测正确的样本数量。这个值随后被用于计算该类别的准确率。
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        # 计算实例准确率：
        correct = pred_choice.eq(target.long().data).cpu().sum() #计算当前批次中正确分类的样本数量。
        mean_correct.append(correct.item() / float(points.size()[0])) # 将当前批次的正确率添加到mean_correct列表中。（8/24，10/24，。。
    '''所有测试批次'''
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]# 计算每个类别的准确率，该类正确预测数量除以该类总数量
    class_acc = np.mean(class_acc[:, 2])#所有类别的平均准确率。
    instance_acc = np.mean(mean_correct)#所有批次的平均实例准确率。（8/24，10/24。。）/103（ni/batch_size）/i  i=测试样本数/batch_size
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'
    #训练集：9843个点云样本（不是点）测试集：2468个点云样本 
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    #dataloader生成训练和测试的实例 shuffle = true 对数据集打乱 num_worker = 4 线程个数为4
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)#24

    '''MODEL LOADING'''
    num_class = args.num_category #分类类别数目
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try: #看是否有训练好的网络模型
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam': #创建一个adam的优化器
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9) #SGD随机梯度下降法

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        #optimizer.step()通常在每个mini-batch之中，而scheduler.step（）通常用在epoch里面，
        #但也不是绝对的，可以根据具体的需求来做。
        #只有用了optimizer.step(),模型才会更新，而scheduler.step()是对lr进行调整。
        scheduler.step()
        # 权重迭代更新410次,每次学到24张点云图的内容
        # 训练集样本量固定，要把训练集样本学完，每学习一次模型梯度更新一次，
        # 而batch_size就是每次学习的样本量，而总训练样本/batch_size=模型权重更新的次数
        '''
        points:torch.Size([12, 1024, 6]) [B,N,D]
        target:torch.Size([12]) 十二张图像的标签
        '''
        '''learning one epoch'''
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # 优化器梯度清零
            optimizer.zero_grad()
            # 数据增强
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # 数据被转换为PyTorch的Tensor格式，并移动到GPU上加速计算（如果指定）。
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            # 模型前向传播：将处理后的点云数据points输入到模型classifier中，得到预测结果pred和全局特征trans_feat。
            pred, trans_feat = classifier(points)
            # 计算损失：使用损失函数criterion计算模型预测结果pred与真实标签target之间的损失。
            loss = criterion(pred, target.long(), trans_feat)
            # 计算预测结果：在各预测值中找出每个样本预测概率最大的类别作为预测结果       
            pred_choice = pred.data.max(1)[1]  
            # 计算该batch中（24个）正确分类数量
            correct = pred_choice.eq(target.long().data).cpu().sum() # 判断预测值是否和target相等，相等则分类正确
            mean_correct.append(correct.item() / float(points.size()[0]))# 计算正确分类数量均值 [12/24,18/24,...
            loss.backward() #反向传播（梯度计算）
            optimizer.step() #更新权重
            global_step += 1
        # 所有数据学习过一遍的迭代梯度更新后的acc均值
        train_instance_acc = np.mean(mean_correct) 
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        #性能评估，梯度不需要更新 每一个epoch
        with torch.no_grad():
            # 测试集样本量固定，要把测试集样本测试完，测试103次instance_acc, class_acc更新，
            # 而batch_size就是每次测试的样本量，而总测试样本/batch_size=模型权重更新的次数
            '''
            points:torch.Size([12, 1024, 6]) [B,N,D]
            target:torch.Size([12]) 十二张图像的标签
            '''
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                #保存网络模型
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
