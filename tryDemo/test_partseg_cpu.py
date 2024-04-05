"test for cpu"
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # 字典{0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes): # num_classes = 16
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--use_cpu', action='store_true', default=True, help='use cpu mode')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def generate_predict_to_txt():
    # 测试不需要权重更新
    with torch.no_grad():
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        # 对testloader中的points进行预测 
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            #print("point.shape: ", points.shape) # point.shape:  torch.Size([24, 2048, 6])
            #points, label, target = points.float(), label.long(), target.long()

            points = points.transpose(2, 1)
            xyz_feature_point = points[:, :6, :]

            seg_pred, _ = classifier(points, to_categorical(label, num_classes)) #推理
            seg_pred = seg_pred.cpu().data.numpy()

            if self.heat_map:
                out =  np.asarray(np.sum(seg_pred,axis=2))
                seg_pred = ((out - np.min(out) / (np.max(out) - np.min(out))))
            else:
                seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c

            seg_pred = np.concatenate([np.asarray(xyz_feature_point), seg_pred[:, None, :]],
                    axis=1).transpose((0, 2, 1)).squeeze(0)
            

def draw_each_img(self,root,idx,name=None,skip=1,save_path=None,heat_maps=False):
    "root：每个txt文件的路径"
    points = np.loadtxt(root)[:, :3]  # 点云的xyz坐标
    points_all = np.loadtxt(root)  # 点云的所有坐标
    points = self.pc_normalize(points)
    skip = skip  # Skip every n points

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    point_range = range(0, points.shape[0], skip)  # skip points to prevent crash
    #print("point_range: ", point_range) #range(0, 2048)
    x = points[point_range, 0]
    z = points[point_range, 1]
    y = points[point_range, 2]

    "根据传入的类别数 自定义生成染色板  标签 0对应 随机颜色1  标签1 对应随机颜色2"
    if self.color_map is not None:
        color_map = self.color_map
    else:
        color_map  = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_classes))} #50
    Label = points_all[point_range, -1] # 拿到标签 取(0~2048)行,-1表示选择每行的最后一个元素
    # 将标签传入前面的字典，找到对应的颜色 并放入列表
    Color = list(map(lambda x: color_map[x], Label))
    ax.scatter(x,  # x
                y,  # y
                z,  # z
                c=Color,  # Color,  # height data for color
                s=25,
                marker=".")
    ax.axis('auto')  # {equal, scaled}
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('off')  # 设置坐标轴不可见
    ax.grid(False)  # 设置背景网格不可见
    ax.view_init(elev=0, azim=0)

    if save_path is None:
        plt.savefig(os.path.join(self.label_path_3d_img,f'{idx}_label_img.png'), dpi=300,bbox_inches='tight',transparent=True)
    else:
        plt.savefig(os.path.join(save_path, f'{idx}_{name}_img.png'), dpi=300, bbox_inches='tight',
                    transparent=True)

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal)
    device = torch.device('cpu')
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location = device)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    
            
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
