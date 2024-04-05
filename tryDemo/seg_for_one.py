"seg for one successful by supported weights file"
import tqdm
import matplotlib
import torch
import os
import warnings
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
#import pybullet as p
from models.pointnet2_part_seg_msg import get_model as pointnet2
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
matplotlib.use("Agg")
IMG_ROOT = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/modelnet40_normal_resampled/airplane/airplane_0001.txt'
#IMG_ROOT = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03642806/1a46d6683450f2dd46c0b76a60ee4644.txt'

folder_name = os.path.dirname(IMG_ROOT).split('/')[-2]  # 获取倒数第二个目录名  
if folder_name == 'modelnet40_normal_resampled':
    d = ','
else:
    d = ' ' 
NUM_CLASSES = 16
DATA = np.loadtxt(IMG_ROOT, delimiter = d).astype(np.float32)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self, img_root, target_root, npoints, normal_channel=False):
        self.npoints = npoints # 采样点数
        self.cat = {}
        self.img_root = img_root
        self.target_root = target_root
        self.normal_channel = normal_channel # 是否使用法向信息
        
        self.cat = {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627',
        'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806',
        'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987',
        'Table': '04379243'} # 16个打雷对应的文件名catalog

        self.classes = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8,
        'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

    def __getitem__(self, index):
        cat = list(self.cat.keys())[0] # list[0] = 'Airplane
        cls = self.classes[cat] # 将类名转换为索引0
        cls = np.array([cls]).astype(np.int32) # [0]
        data = DATA
        if not self.normal_channel:  # 判断是否使用法向信息
            self.point_set = data[:, 0:3]
        else:
            self.point_set = data[:, 0:6]
        seg = data[:, -1].astype(np.int32)# 预留出来拿到小类别的标签[0 0 0 ... 0 0 0]全部点的类别信息

        self.point_set[:, 0:3] = pc_normalize(self.point_set[:, 0:3]) # 做一个归一化
        #choice = np.random.choice(self.point_set.shape[0], self.npoints, replace=True) # 对一个类别中的数据进行随机采样 返回索引，允许重复采样
        choice = np.random.choice(len(seg), self.npoints, replace=True) # 对一个类别中的数据进行随机采样 返回索引，允许重复采样
        # resample
        self.point_set =  self.point_set[choice, :] # 根据索引采样
        seg = seg[choice] #newdata = self.raw_pcd # data中存放txt中所有点的信息
        return self.point_set, cls, seg # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别

    def __len__(self):
        return 1

class Generate_txt_and_3d_img:
    def __init__(self,img_root,target_root,num_parts,testDataLoader,model,visualize = False, color_map=None):
        self.img_root = img_root # 点云数据路径
        self.target_root = target_root  # 生成txt标签和预测结果路径
        self.testDataLoader = testDataLoader
        self.num_parts = num_parts
        self.heat_map = False # 控制是否输出heatmap
        self.visualize = visualize # 是否open3d可视化
        self.color_map = color_map
        
        self.model = model
        self.model_name = 'pointnet2_part_seg_msg'
        # 创建文件夹
        self.all_pred_image_path = "" # 所有预测结果的路径列表
        self.all_pred_txt_path = "" # 所有预测txt的路径列表

        self.all_pred_txt_path = os.path.join(self.target_root,'_predict_txt')
        self.make_dir(self.all_pred_txt_path)

        self.all_pred_image_path = os.path.join(self.target_root, '_predict_image')
        self.make_dir(self.all_pred_image_path)
        
        self.label_path_txt = os.path.join(self.target_root, 'label_txt') # 存放label的txt文件
        self.make_dir(self.label_path_txt)

        self.label_path_3d_img = os.path.join(self.target_root, 'label_3d_img')
        self.make_dir(self.label_path_3d_img)

        self.generate_predict()
        self.draw_3d_img()

    def generate_predict(self): #target real label
        for batch_id, (points, label, target) in tqdm.tqdm(enumerate(self.testDataLoader),
                                                                      total=len(self.testDataLoader),smoothing=0.9):
            #点云数据、整个图像的标签、每个点的标签、 没有归一化的点云数据（带标签）torch.Size([1, 7, 2048])
            """
            before points.shape:  torch.Size([1, 2048, 6])
            label.shape:  torch.Size([1, 1])
            target.shape:  torch.Size([1, 2048])
            after points.shape:  torch.Size([1, 6, 2048])
            point_set_without_normal.shape:  (2048, 7)
            """
            print("label: ",label)
            print("target: ",target)
            points = points.transpose(2, 1)
            # torch.Size([1, 6, 2048]) 有一个txt文件，每个文件中有2048个点，每个点包含六个维度的信息
           
            xyz_feature_point = points[:, :6, :]
            """
            points.permute(0, 2, 1):参数指定了新的维度顺序。
            将张量的维度从(batch_size, num_points, num_channels)变为(batch_size, num_channels, num_points)。
            target[:,:,None]: 这部分代码取名为target的张量，并在最后一个维度上增加一个新的维度。这通常是为了与points的维度匹配，以便于连接操作。
            .squeeze(0): 这将删除数组的第一个维度（如果它的大小为1）
            """
            # 将points和target两个张量连接在一起，并转换为NumPy数组。
            point_set_without_normal = np.asarray(torch.cat([points.permute(0, 2, 1),target[:,:,None]],dim=-1)).squeeze(0)  # 代标签 没有归一化的点云数据  的numpy形式
            np.savetxt(os.path.join(self.label_path_txt,f'{batch_id}_label.txt'), point_set_without_normal, fmt='%.04f') # 将其存储为txt文件

            seg_pred, _ = self.model(points, self.to_categorical(label, NUM_CLASSES)) #推理
            seg_pred = seg_pred.cpu().data.numpy()

            if self.heat_map:
                out =  np.asarray(np.sum(seg_pred,axis=2))
                seg_pred = ((out - np.min(out) / (np.max(out) - np.min(out))))
            else:
                seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c

            # 将点云与预测结果进行拼接，准备生成txt文件
            seg_pred = np.concatenate([np.asarray(xyz_feature_point), seg_pred[:, None, :]],
                    axis=1).transpose((0, 2, 1)).squeeze(0)  
           
            save_path = os.path.join(self.all_pred_txt_path, f'{self.model_name}_{batch_id}.txt')
            np.savetxt(save_path, seg_pred, fmt='%.04f') # 路径 点云与预测结果拼接后的txt文本 
            # 将numpy数组转换为字符串并保存到文件中  # 意味着保存的浮点数将保留小数点后四位

    def draw_3d_img(self):
        # 调用matpltlib 画3d图像
        #用于列出指定目录下的所有文件和子目录的名称。
        each_txt_path = os.listdir(self.all_pred_txt_path) # 拿到txt文件的全部名字_predict_txt
        each_label = os.listdir(self.label_path_txt)  # 所有标签txt路径  'label_txt'

        pre_txt_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result/_predict_txt'
        save_img_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result/_predict_image'
        for idx,(txt,lab) in tqdm.tqdm(enumerate(zip(each_txt_path,each_label)),total=len(each_txt_path)):
            self.draw_each_img(os.path.join(self.label_path_txt, lab), idx, heat_maps=False) # self.label_path_txt: result/label_txt
            self.draw_each_img(os.path.join(pre_txt_path,txt), idx, save_path=save_img_path, heat_maps=self.heat_map)
            # pre_txt_path = pointnet_predict_txt'save_img_path = pointnet_predict_image'
        print(f'所有预测图片已生成完毕，请前往：{self.all_pred_image_path} 查看')

    def draw_each_img(self,root,idx,name=None,skip=1,save_path=None,heat_maps=False):
        "root：每个txt文件的路径"
        points = np.loadtxt(root)[:, :3]  # 点云的xyz坐标
        points_all = np.loadtxt(root)  # 点云的所有坐标
        points = pc_normalize(points)
        skip = skip  # Skip every n points

        # 创建一个新的图形窗口 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #ax = plt.subplot(111, projection='3d')

        point_range = range(0, points.shape[0], skip)  # skip points to prevent crash
        #print("point_range: ", point_range) #range(0, 2048)
        x = points[point_range, 0]
        z = points[point_range, 1]
        y = points[point_range, 2]

        "根据传入的类别数 自定义生成染色板  标签 0对应 随机颜色1  标签1 对应随机颜色2"
        if self.color_map is not None:
            color_map = self.color_map
        else:
            color_map  = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_parts))} #50
        #print("color_map: ", color_map)
        """
        color_map
        {0: 0.0, 1: 0.018367346938775512, 2: 0.036734693877551024, 3: 0.05510204081632654, 4: 0.07346938775510205, 
         5: 0.09183673469387756, 6: 0.11020408163265308, 7: 0.1285714285714286, 8: 0.1469387755102041, 9: 0.1653061224489796, 
         10: 0.1836734693877551, 11: 0.20204081632653062, 12: 0.22040816326530616, 13: 0.23877551020408166, 14: 0.2571428571428572, 
         15: 0.2755102040816327, 16: 0.2938775510204082, 17: 0.3122448979591837, 18: 0.3306122448979592, 19: 0.3489795918367347, 
         20: 0.3673469387755102, 21: 0.38571428571428573, 22: 0.40408163265306124, 23: 0.42244897959183675, 24: 0.4408163265306123, 
         25: 0.4591836734693878, 26: 0.47755102040816333, 27: 0.49591836734693884, 28: 0.5142857142857143, 29: 0.5326530612244899, 
         30: 0.5510204081632654, 31: 0.5693877551020409, 32: 0.5877551020408164, 33: 0.6061224489795919, 34: 0.6244897959183674, 
         35: 0.6428571428571429, 36: 0.6612244897959184, 37: 0.6795918367346939, 38: 0.6979591836734694, 39: 0.7163265306122449, 
         40: 0.7346938775510204, 41: 0.753061224489796, 42: 0.7714285714285715, 43: 0.789795918367347, 44: 0.8081632653061225, 
         45: 0.826530612244898, 46: 0.8448979591836735, 47: 0.863265306122449, 48: 0.8816326530612246, 49: 0.9}
        """
        #color_map:  {0: 0.0, 1: 0.018367346938775512, 2: 0.036734693877551024, 3: 0.05510204081632654, 4: 0.07346938775510205, 5: 0.09183673469387756, 6: 0.11020408163265308, 7: 0.1285714285714286, 8: 0.1469387755102041, 9: 0.1653061224489796, 10: 0.1836734693877551, 11: 0.20204081632653062, 12: 0.22040816326530616, 13: 0.23877551020408166, 14: 0.2571428571428572, 15: 0.2755102040816327, 16: 0.2938775510204082, 17: 0.3122448979591837, 18: 0.3306122448979592, 19: 0.3489795918367347, 20: 0.3673469387755102, 21: 0.38571428571428573, 22: 0.40408163265306124, 23: 0.42244897959183675, 24: 0.4408163265306123, 25: 0.4591836734693878, 26: 0.47755102040816333, 27: 0.49591836734693884, 28: 0.5142857142857143, 29: 0.5326530612244899, 30: 0.5510204081632654, 31: 0.5693877551020409, 32: 0.5877551020408164, 33: 0.6061224489795919, 34: 0.6244897959183674, 35: 0.6428571428571429, 36: 0.6612244897959184, 37: 0.6795918367346939, 38: 0.6979591836734694, 39: 0.7163265306122449, 40: 0.7346938775510204, 41: 0.753061224489796, 42: 0.7714285714285715, 43: 0.789795918367347, 44: 0.8081632653061225, 45: 0.826530612244898, 46: 0.8448979591836735, 47: 0.863265306122449, 48: 0.8816326530612246, 49: 0.9}

        # Label (0~49) 2048个 #Label:  [0. 0. 0. ... 0. 0. 0.]
        Label = points_all[point_range, -1] # 拿到标签 取(0~2048)行,-1表示选择每行的最后一个元素
        
        # 将标签传入前面的字典，找到对应的颜色 并放入列表
        Color = list(map(lambda x: color_map[x], Label)) 
        #Color = list(map(lambda x: print("x: ", x, "  color_map[x]: ",color_map[x]), Label))
        #print("color: ",Color)
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
        #plt.draw() # 更新matplotlib的画布，以便显示刚刚添加到其中的内容

        if save_path is None:
            plt.savefig(os.path.join(self.label_path_3d_img,f'{idx}_label_img.png'), dpi=300,bbox_inches='tight',transparent=True)
        else:
            plt.savefig(os.path.join(save_path, f'{idx}_{name}_img.png'), dpi=300, bbox_inches='tight',
                        transparent=True)
        #plt.show() #用于显示图形
    
    def make_dir(self, root):
        if os.path.exists(root):
            print(f'{root} 路径已存在 无需创建')
        else:
            os.mkdir(root)

    def to_categorical(self,y, NUM_CLASSES):
        """ 1-hot encodes a tensor """
        new_y = torch.eye(NUM_CLASSES)[y.cpu().data.numpy(),]
        if (y.is_cuda):
            return new_y.cuda()
        return new_y

def load_models(model_dict={'PonintNet': [pointnet2(num_classes=50,normal_channel=True).eval(),
                                          r'./log/part_seg/pointnet2_part_seg_msg/checkpoints']}):
    model = list(model_dict.values())[0][0]
    checkpoints_dir = list(model_dict.values())[0][1]
    device = torch.device('cpu')
    weight_dict = torch.load(os.path.join(checkpoints_dir,'best_model.pth'),  map_location = device)
    model.load_state_dict(weight_dict['model_state_dict'])
    return model

"""
class Open3dVisualizer():

	def __init__(self):

		self.point_cloud = o3d.geometry.PointCloud()
		self.o3d_started = False

		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window()

	def __call__(self, points, colors):

		self.update(points, colors)

		return False

	def update(self, points, colors):
		coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.15, origin = [0,0,0])
		self.point_cloud.points = points
		self.point_cloud.colors = colors
		# self.point_cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
		# self.vis.clear_geometries()
		# Add geometries if it is the first time
		if not self.o3d_started:
			self.vis.add_geometry(self.point_cloud)
			self.vis.add_geometry(coord_mesh)
			self.o3d_started = True

		else:
			self.vis.update_geometry(self.point_cloud)
			self.vis.update_geometry(coord_mesh)

		self.vis.poll_events()
		self.vis.update_renderer()
"""

if __name__ =='__main__':

    num_parts = 50 # 填写数据集的类别数 如果是s3dis这里就填13   shapenet这里就填50

    img_root = IMG_ROOT
    target_root = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result' # 输出结果路径
    #print("type(img_root): ", type(img_root))
    #TEST_DATASET 一个txt点云文件的 point_set,cls,seg
    TEST_DATASET = PartNormalDataset(img_root, target_root, npoints=2048, normal_channel=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0,drop_last=True)
    predict_pcd = Generate_txt_and_3d_img(img_root,target_root, num_parts,testDataLoader,load_models(),visualize = True)

    #就是数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。
    #在训练模型时使用到此函数，用来把训练数据分成多个小组，此函数每次抛出一组数据。直至把所有的数据都抛出。就是做一个数据的初始化。