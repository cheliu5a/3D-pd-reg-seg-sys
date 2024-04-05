"cls for one by supported weights file successful"
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import matplotlib.pyplot as plt
import torch.nn.functional as F  
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
LABEL_MAPPING = {
    0:"airplane",
    1:"bathtub",
    2:"bed",
    3:"bench",
    4:"bookshelf",
    5:"bottle",
    6:"bowl",
    7:"car",
    8:"chair",
    9:"cone",
    10:"cup",
    11:"curtain",
    12:"desk",
    13:"door",
    14:"dresser",
    15:"flower_pot",
    16:"glass_box",
    17:"guitar",
    18:"keyboard",
    19:"lamp",
    20:"laptop",
    21:"mantel",
    22:"monitor",
    23:"night_stand",
    24:"person",
    25:"piano",
    26:"plant",
    27:"radio",
    28:"range_hood",
    29:"sink",
    30:"sofa",
    31:"stairs",
    32:"stool",
    33:"table",
    34:"tent",
    35:"toilet",
    36:"tv_stand",
    37:"vase",
    38:"wardrobe",
    39:"xbox"
}
 
def pc_normalize(pc):  #点云数据归一化
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=True, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=10000, help='Point Number')
    #parser.add_argument('--log_dir', type=str, default='pointnet2_cls_ssg_normal', help='Experiment root')
    parser.add_argument('--log_dir', type=str, default='pointnet2_msg_normals', help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    print(parser.parse_args())
    return parser.parse_args()
#加载数据集 对数据集进行预处理
dataset='/home/chen/Pointnet_Pointnet2_pytorch-master/data/modelnet40_normal_resampled/cup/cup_0001.txt'
#dataset='/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1a04e3eab45ca15dd86060f189eb133.txt'
print(type(dataset))
#pcdataset = np.loadtxt(dataset, delimiter=' ').astype(np.float32) # shapenet
pcdataset = np.loadtxt(dataset, delimiter=',').astype(np.float32)# modelnet40数据读取，我的数据是三（6）个维度，数据之间是空格，如果是逗号修改一下即可
print(type(pcdataset))
point_set = pcdataset[0:10000, :] #我的输入数据设置为原始数据中10000个点 model40 有10000个点的数据
#point_set = pcdataset[0:2048, :] # shapenet 2800左右数据
point_set[:, 0:6] = pc_normalize(point_set[:, 0:6]) #归一化数据
point_set = point_set[:, 0:6] # : 表示选择所有的行。0:6 表示从第0列到第6列（不包括第6列）选择列。
point_set = point_set.transpose(1,0)#将数据由N*C转换为C*N
print("point_set.shape: ", point_set.shape)

point_set = point_set.reshape(1, 6, 10000)
#point_set = point_set.reshape(1, 6, 2048)
n_points = point_set
point_set = torch.as_tensor(point_set) # 需要将数据格式变为张量，不然会报错

def percent(num):
    # 转换为小数  
    val = float(num)  
    
    # 转换为百分比  
    percentage = val * 100  
    
    res =  f"{percentage:.2f}%"

    # 输出结果  
    print(res) 
    return res
'''
97.47%
0.65%
0.37%
'''
#分类测试函数
def test(model,point_set, num_class=40, vote_num=1):
    classifier = model.eval()
    #class_acc = np.zeros((num_class, 3))
    vote_pool = torch.zeros(1, num_class)

    for _ in range(vote_num):
        pred, _ = classifier(point_set)
        #print(pred) # 3次 num_votes=3
        vote_pool += pred
    pred = vote_pool / vote_num
    
    # 在pred中取前三个大
    # print("pred: ", pred)
    # print("pred.shape: ", pred.shape)
    '''
         tensor([[ -9.8474,  -9.0500,  -9.8747,  -9.8886, -14.7912,  -6.8798,  -4.6940,
         -10.0690,  -7.0668,  -6.8531,  -0.0836,  -5.6331, -13.1227,  -8.7341,
         -11.1858,  -4.8096,  -7.0476,  -6.3485,  -8.0844,  -5.7002,  -7.9230,
         -14.0615,  -6.0359,  -8.8494,  -9.6227, -11.0600,  -7.3424,  -5.5257,
         -11.7071,  -9.9881, -11.9145,  -9.0035,  -7.8445,  -7.3272,  -6.0156,
         -10.1473, -11.7695,  -3.9522, -10.1107,  -8.6604]])
         pred.shape:  torch.Size([1, 40])

    '''
    # 如果你想要查看概率分布，可以使用 softmax 函数  
    # 对预测结果每行取最大值得到分类
    pred_choice = pred.data.max(1)[1]
    '''
    score, top3_indices = pred.topk(3, dim=1)  
    print("top3_indices:", top3_indices) # tensor([[10, 37,  6]]) torch.Size([1, 3]) 
    print("score: ", score)
    # 获取第一个值  
    top1 = top3_indices[0, 0].item()  # 结果是 10  
    
    # 获取第二个值  
    top2 = top3_indices[0, 1].item()  # 结果是 37  
    
    # 获取第三个值  
    top3 = top3_indices[0, 2].item()  # 结果是 6
    '''
    #查看概率分布，可以使用 softmax 函数  
    probabilities = F.softmax(pred, dim=1)  
    # 输出概率分布  
    #print(f"概率分布是：\n{probabilities.squeeze()}")  
    '''
    概率分布是：
tensor([2.7516e-05, 4.8191e-05, 1.5314e-05, 1.4114e-05, 1.1886e-07, 6.0469e-04,
        3.3261e-03, 1.3240e-05, 3.5237e-04, 5.3571e-04, 9.6860e-01, 1.6738e-03,
        5.6082e-07, 8.4091e-05, 6.0778e-06, 3.9739e-03, 2.8110e-04, 9.8400e-04,
        9.7906e-05, 2.7498e-03, 1.2882e-04, 2.6522e-07, 1.8098e-03, 5.8715e-05,
        3.5586e-05, 4.5270e-06, 2.5452e-04, 1.7620e-03, 2.8483e-06, 1.9049e-05,
        1.6650e-06, 7.5067e-05, 3.0545e-04, 2.4337e-04, 8.0669e-04, 2.4113e-05,
        1.7652e-06, 1.0966e-02, 1.6958e-05, 9.6469e-05])
    '''
    # 使用 topk 方法获取前三个最大值的索引  
    # 注意：topk 返回的是值和索引的元组 
    score, top3_indices = probabilities.topk(3, dim=1)  

    #print("top3_indices: ", top3_indices, "score: ", score)

    # 获取第一个索引  
    top1_id = top3_indices[0, 0].item()  # 结果是 10
    
    # 获取第二个值  
    top2_id = top3_indices[0, 1].item()  # 结果是 37
    
    # 获取第三个值  
    top3_id = top3_indices[0, 2].item()  # 结果是 6

    # 获取第一个分数  
    top1_score = score[0, 0].item()  # 结果是 10
    
    # 获取第二个值  
    top2_score = score[0, 1].item()  # 结果是 37
    
    # 获取第三个值  
    top3_score = score[0, 2].item()  # 结果是 6
    
    percent(top1_score)
    percent(top2_score)
    percent(top3_score)

    '''
    print('top1_score: ', top1_score)
    print('top2_score: ', top2_score)
    print('top3_score: ', top3_score)
    
    top1: tensor(10) top2:  tensor(37) top3:  tensor(15)
    print("top1:", top1, "top2: ", top2, "top3: ", top3)
    print("pred_choice: ", pred_choice) # tensor([num]) tensor([10])
    print("pred_choice.item(): ", pred_choice.item()) # num 10
    print("label: ",LABEL_MAPPING[pred_choice.item()]) # cup
    '''
    #print("top1:", top1, "top2: ",top2, "top3: ", top3)
    #可视化
    file_dir = '/home/chen/Pointnet_Pointnet2_pytorch-master/visualizer'
    save_name_prefix = 'pred'
    img_path = draw(n_points[:, 0, :], n_points[:, 1, :], n_points[:, 2, :], save_name_prefix, file_dir, color=pred_choice)
    return img_path, pred_choice 

#定义可视化函数
def draw(x, y, z, name, file_dir, color=None): # name:pred save_name 图片保存路径
    """
    绘制单个样本的三维点图
    """
    if color is None:
        for i in range(len(x)):
            ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
            save_name = name + '-{}.png'.format(i)
            save_name = os.path.join(file_dir,save_name)
            ax.scatter(x[i], y[i], z[i],s=0.1, c='r')
            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.draw() # 更新matplotlib的画布，以便显示刚刚添加到其中的内容
            plt.savefig(save_name)
            plt.show() #用于显示图形
    else:
        colors = [
            'red', 'blue', 'green', 'yellow', 'orange', 'tan', 'orangered', 'lightgreen', 'coral', 'aqua',
            'pink', 'dodgerblue', 'olive', 'aquamarine', 'azure', 'beige', 'bisque', 'violet', 'blueviolet','brown',
            'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
            'darkgoldenrod', 'dimgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon'
            ]
        for i in range(len(x)):
            #print(len(x)) #len(x) = 1
            #print(i) #0
            ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程

            save_name = name + '-{}-{}.png'.format(i, color[i])
            save_name = os.path.join(file_dir,save_name)
            ax.scatter(x[i], y[i], z[i],s=0.1, c=colors[color[i]])
            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.draw()
            plt.savefig(save_name)
            plt.show()
        return save_name
 
def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    '''CREATE DIR'''
    experiment_dir = '/home/chen/Pointnet_Pointnet2_pytorch-master/log/classification/' + args.log_dir
    args = parse_args()

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    
    #选择训练好的.pth文件
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location = device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    #预测分类
    with torch.no_grad():
        pred_choice = test(classifier.eval(), point_set, vote_num=args.num_votes, num_class=num_class)
        
if __name__ == '__main__':
    args = parse_args()
    main(args)