import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

#点云归一化，以centroid为中心，半径为1
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

#最远点采样函数
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point 
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            #rstrip()删除string字符串末尾的指定字符
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index] #从self.datapath中获取点云数据，格式为（类别名称',路径）
            cls = self.classes[self.datapath[index][0]]# 根据索引和类别名称的对应关系，将名称与索引对应起来，例如airplane：0
            label = np.array([cls]).astype(np.int32)#转换为数组模式
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)#使用np读点云数据 点云数据结果为10000*6
            #数据集采样npoints个点送入网络
            if self.uniform:# 默认为False，没有使用FPS算法筛选，降采样到self.npoints
                point_set = farthest_point_sample(point_set, self.npoints)
            else: #取前1024个数
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) #只对点的前三个维度进行归一化，即坐标的归一化
        if not self.use_normals: # 不使用normals信息，只需返回坐标维度
            point_set = point_set[:, 0:3]

        return point_set, label[0] # 返回读取到的经过降采样的点云数据和标签
        '''
        torch.Size([12, 1024, 6]) [B,N,D]
        torch.Size([12]) 十二张图像的标签 BATCH_SIZE:即一次训练所抓取的数据样本数量12张图的标签
        '''

    def __getitem__(self, index):
        return self._get_item(index)

'''
if __name__ == '__main__':
    import torch
    
    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
'''

if __name__ == '__main__':
    import torch
    import argparse

    def parse_args():
        '''PARAMETERS'''
        parser = argparse.ArgumentParser('training')
        parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
        parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
        parser.add_argument('--model', default='pointnet2_msg_normals', help='model name [default: pointnet_cls]')
        parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
        parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
        parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
        parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
        parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
        parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
        parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate') #权重衰减10^-4
        parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
        parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
        parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')#默认不适用FPS
        return parser.parse_args()

    args = parse_args()
    root = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/modelnet40_normal_resampled'
    data = ModelNetDataLoader(root = root, args = args, split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
        '''
        torch.Size([12, 1024, 6]) [B,N,D]
        torch.Size([12]) 十二张图像的标签
        '''