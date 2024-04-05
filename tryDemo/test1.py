from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import os
import sys
import importlib
import numpy as np
import argparse
import torch
import matplotlib
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QLabel, QFileDialog, QTextEdit, QFormLayout, QMessageBox
from PyQt5.QtGui import QTextCursor  
from PyQt5.QtGui import QPixmap, QFont  
from PyQt5.QtCore import QRect
import tqdm
import warnings
from torch.utils.data import Dataset
import pybullet as p
from models.pointnet2_part_seg_msg import get_model as pointnet2

warnings.filterwarnings('ignore')
matplotlib.use("Agg")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
LABEL_MAPPING = {
    0:"Airplane",
    1:"Bathtub",
    2:"Bed",
    3:"Bench",
    4:"Bookshelf",
    5:"Bottle",
    6:"Bowl",
    7:"Car",
    8:"Chair",
    9:"Cone",
    10:"Cup",
    11:"Curtain",
    12:"Desk",
    13:"Door",
    14:"Dresser",
    15:"Flower_pot",
    16:"Glass_box",
    17:"Guitar",
    18:"Keyboard",
    19:"Lamp",
    20:"Laptop",
    21:"Mantel",
    22:"Monitor",
    23:"Night_stand",
    24:"Person",
    25:"Piano",
    26:"Plant",
    27:"Radio",
    28:"Range_hood",
    29:"Sink",
    30:"Sofa",
    31:"Stairs",
    32:"Stool",
    33:"Table",
    34:"Tent",
    35:"Toilet",
    36:"Tv_stand",
    37:"Vase",
    38:"Wardrobe",
    39:"Xbox"
}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NOTCHOOSED = 0
CHOOSED = 1
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def percent(num):
    # 转换为小数  
    val = float(num)  
    # 转换为百分比  
    percentage = val * 100  
    res =  f"{percentage:.2f}%"
    # 输出结果  
    #print(res) 
    return res

class Child1Window(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('物体识别窗口')
       # 窗体的尺寸
        self.resize(900, 783)

        self.pre_cls = None
        
        # 窗体位置
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)

        # 创建布局 垂直方向
        container = QVBoxLayout()

        container.addLayout(self.init_header()) # 在布局中添加header button

        container.addLayout(self.init_form())  # 在布局中添加识别结果文本

        container.addLayout(self.init_title())  # 在布局中添加历史记录
        container.addLayout(self.init_table())  # 在布局中添加表格对象

        container.addLayout(self.init_label())  # 在布局中添加标签
        container.addLayout(self.init_show()) # 在布局中显示图像
        
        # 弹簧 在四个元素的最下面加弹簧，把元素顶上去
        #layout.addStretch()
        # 给窗体设置元素的排列方式
        self.setLayout(container)
 
    def parse_args(self):
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
        parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting') #投票数均值
        #print(parser.parse_args())
        return parser.parse_args()
    
    def init_header(self):
        # 1.创建顶部菜单布局
        header_layout = QHBoxLayout() # 创建对象
        # 1.1 创建按钮，加入header_layout
        btn_open_file = QPushButton('打开文件')
        btn_open_file.clicked.connect(self.openFile)
        header_layout.addWidget(btn_open_file)
        btn_cls = QPushButton('开始识别')
        btn_cls.clicked.connect(self.button_cls_click)
        header_layout.addWidget(btn_cls)
        btn_check = QPushButton('查看点云图片')
        btn_check.clicked.connect(self.button_check_click)
        header_layout.addWidget(btn_check)
        # 弹簧把元素弹到左侧
        header_layout.addStretch()
        return header_layout

    def init_form(self):
        # 表单容器
        form_layout = QFormLayout()
        # 创建1个输入框
        self.edit = edit = QLineEdit()
        #edit.setFixedWidth(200)
        edit.setPlaceholderText("")
        form_layout.addRow("识别结果：", edit)
        # 创建另外1个输入框
        self.edit2 = edit2 = QLineEdit()
        #edit2.setFixedWidth(200)
        edit2.setPlaceholderText("")
        form_layout.addRow("识别分数：", edit2)
        # 将from_layout添加到垂直布局器中
        return form_layout
    
    def init_title(self):
        label_layout = QHBoxLayout() # 创建标签对象
        label_layout.addStretch(1)
        title = QLabel('历史记录')
        label_layout.addWidget(title)
        label_layout.addStretch(1)
        return label_layout
    
    def init_table(self):
        # 3.创建中间表格布局
        table_layout = QHBoxLayout()  # 创建对象

        # 3.1 创建表格
        self.table_widget = table_widget = QTableWidget(0,3) # 默认显示0行2列
        self.table_widget.setFixedSize(900, 300) # 默认显示0行2列
        table_widget.setObjectName("tableWidget")
        item = QTableWidgetItem()
        item.setText("文件名") 
        table_widget.setHorizontalHeaderItem(0,item) # 设置横向标题名字
        table_widget.setColumnWidth(0,300)
        item = QTableWidgetItem()
        item.setText("识别结果") 
        table_widget.setHorizontalHeaderItem(1,item) # 设置横向标题名字
        table_widget.setColumnWidth(1,300)
        item = QTableWidgetItem()
        item.setText("识别分数") 
        table_widget.setHorizontalHeaderItem(2,item) # 设置横向标题名字
        table_widget.setColumnWidth(2,300)
        table_layout.addWidget(table_widget)
        return table_layout
        
    def init_label(self):
        label_layout = QHBoxLayout() # 创建标签对象
        label_layout.addStretch(1)
        label1 = QLabel('三维点云图')
        label_layout.addWidget(label1)
        label_layout.addStretch(1)
        return label_layout
    
    def init_show(self):
        label_layout = QHBoxLayout() # 创建标签对象
        self.label3 = label3 = QLabel()
        label3.setFixedSize(400, 400)
        label3.setText("")
        label_layout.addWidget(label3)
        return label_layout
    
    def openFile(self): 
        """
        打开文件，并把文件名显示到表格
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*);;Text Files (*.txt)", options=options)
        if fileName == '':
            QMessageBox.warning(self,"错误","您必需选择一个文件")
            self.status = NOTCHOOSED
            #print("# ",self.status)
            return
        self.status = CHOOSED
        self.fileName = fileName
        #print("文件已打开")
        #获取文件名，显示到表格
        #print(fileName)
        self.name = name = os.path.basename(fileName)
        #print(name)

        # cell = QTableWidgetItem(str(name)) # 获取文件名
        # self.table_widget.setItem(0,0,cell) # 行，列放cell #传递参数，只要在定义她的时候加self line：50

        cell = QTableWidgetItem(str(name)) # 获取文件名
        self.current_row_count = current_row_count = self.table_widget.rowCount() # 获取当前表格有多少行
        self.table_widget.insertRow(current_row_count)
        #print("当前表格第:",current_row_count)
        self.table_widget.setItem(current_row_count,0,cell) # 行，列放cell #传递参数，只要在定义她的时候加self line：50
        #print("当前表格第:",current_row_count)
        #print("## ",self.status)
        return fileName 
    
    def data_preprocess(self):
        '''数据预处理self.n_points(用于可视化),point_set(用于识别模型)'''
        pcdataset = np.loadtxt(self.fileName, delimiter=',').astype(np.float32)#数据读取，我的数据是三（6）个维度，数据之间是空格，如果是逗号修改一下即可
        point_set = pcdataset[0:10000, :] #我的输入数据设置为原始数据中10000个点
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) #归一化数据
        #point_set[:, 0:6] = pc_normalize(point_set[:, 0:6]) #归一化数据
        point_set = point_set[:, 0:6] 
        point_set = point_set.transpose(1,0)#将数据由N*C转换为C*N
        point_set = point_set.reshape(1, 6, 10000)
        self.n_points = point_set # 赋值给这个类的对象，通过self.n_points传值
        self.point_set = torch.as_tensor(point_set) # 需要将数据格式变为张量，不然会报错

    def button_cls_click(self):
        """
        前提：表格中有文件名。识别文件，并把识别结果显示到表格和文本
        """
        if not self.status:
            QMessageBox.warning(self,"错误","您必需选择一个文件")
            return
        '''
        数据预处理
        pcdataset = np.loadtxt(self.fileName, delimiter=',').astype(np.float32)#数据读取，我的数据是三（6）个维度，数据之间是空格，如果是逗号修改一下即可
        point_set = pcdataset[0:10000, :] #我的输入数据设置为原始数据中10000个点
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) #归一化数据
        #point_set[:, 0:6] = pc_normalize(point_set[:, 0:6]) #归一化数据
        point_set = point_set[:, 0:6] 
        point_set = point_set.transpose(1,0)#将数据由N*C转换为C*N
        point_set = point_set.reshape(1, 6, 10000)
        self.n_points = point_set # 赋值给这个类的对象，通过self.n_points传值
        point_set = torch.as_tensor(point_set) # 需要将数据格式变为张量，不然会报错
        '''
        self.data_preprocess()
        args = self.parse_args()

        '''HYPER PARAMETER'''
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        '''CREATE DIR'''
        experiment_dir = '/home/chen/Pointnet_Pointnet2_pytorch-master/log/classification/' + args.log_dir
        '''LOG'''
        num_class = args.num_category

        #选择模型
        model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
        #print("model name: ", model_name) # pointnet2_cls_ssg
        model = importlib.import_module(model_name)
        classifier = model.get_model(num_class, normal_channel=args.use_normals)
        #选择训练好的.pth文件 
        device = torch.device('cpu')
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location = device) 
        classifier.load_state_dict(checkpoint['model_state_dict'])
        #预测分类
        with torch.no_grad():
            top1_label, top2_label, top3_label, score1, score2, score3 = self.test(classifier.eval(), self.point_set, vote_num=args.num_votes, num_class=num_class)

        self.pre_cls = pre_cls = top1_label
        
        # 将标签显示在表单上
        self.edit.setText(str(pre_cls))
        self.edit2.setText(str(score1))

        # 将标签显示在表格上
        cell = QTableWidgetItem(str(pre_cls))
        current_row_count = self.current_row_count
        #current_row_count = self.table_widget.rowCount() # 获取当前表格有多少行 会使得currow加1
        self.table_widget.setItem(current_row_count,1,cell) # 行，列 放cell
        # 将分数显示在表格上
        cell = QTableWidgetItem(score1)
        current_row_count = self.current_row_count
        self.table_widget.setItem(current_row_count,2,cell) # 行，列 放cell
        self.current_row_count = current_row_count + 1

    def button_check_click(self):
        #可视化
        file_dir = '/home/chen/Pointnet_Pointnet2_pytorch-mastn_pointser/visualizer'
        save_name_prefix = 'pred_lab'
        self.data_preprocess()#使用n_points
        n_points = self.n_points # 通过self.n_points传值
        save_path = self.draw(n_points[:, 0, :], n_points[:, 1, :], n_points[:, 2, :], save_name_prefix, file_dir)
        # 将save_path中的路径图片显示到标签
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        img = QPixmap(save_path).scaled(self.label3.width(), self.label3.height())
        # 在label控件上显示选择的图片
        self.label3.setPixmap(img)
        self.label3.setScaledContents(True)
    
    def test(self, model, point_set, num_class=40, vote_num=1):
        # 计算分类分数和标签
        classifier = model.eval()
        #class_acc = np.zeros((num_class, 3))
        vote_pool = torch.zeros(1, num_class)

        for _ in range(vote_num):
            pred, _ = classifier(point_set)
            #print(pred) # 3次 num_votes=3
            vote_pool += pred
        pred = vote_pool / vote_num
        # 对预测结果每行取最大值得到分类
        #self.pred_choice = pred.data.max(1)[1] # tensor([idx])
        '''
        print("pred: ", pred)
        # 使用 topk 方法获取前三个最大值的索引  
        # 注意：topk 返回的是值和索引的元组  
        _, top3_indices = pred.topk(3, dim=1) #tensor([[idx1,idx2,idx3]])  
        # 获取第一个值  
        top1_id = top3_indices[0, 0].item()  # 结果是 10  
        # 获取第二个值  
        top2_id = top3_indices[0, 1].item()  # 结果是 37  
        # 获取第三个值  
        top3_id = top3_indices[0, 2].item()  # 结果是 6
        '''
        #查看概率分布，可以使用 softmax 函数  
        import torch.nn.functional as F 
        probabilities = F.softmax(pred, dim=1)  
        
        # 输出概率分布  
        #print(f"概率分布是：\n{probabilities.squeeze()}")  
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
        
        score1 = percent(top1_score)
        score2 = percent(top2_score)
        score3 = percent(top3_score)

        return LABEL_MAPPING[top1_id], LABEL_MAPPING[top2_id], LABEL_MAPPING[top3_id], score1, score2, score3#, save_path
    
    def draw(self, x, y, z, name, file_dir): 
        '''
        定义可视化函数
        绘制单个样本的三维点图,返回点云图保存路径
        if color is None:
            for i in range(len(x)):
                ax = plt.subplot(projection='3d') # 创建一个三维的绘图工程
                save_name = name + '-{}.png'.format(i)
                save_name = os.path.join(file_dir,save_name)
                ax.scatter(x[i], y[i], z[i],s=0.1, c='r')
                ax.axis('auto') 
                ax.set_zlabel('Z')  # 坐标轴
                ax.set_ylabel('Y')
                ax.set_xlabel('X')
                #ax.axis('off')  # 设置坐标轴不可见
                #ax.grid(False)  # 设置背景网格不可见
                plt.draw()
                plt.savefig(save_name)
                plt.show()
                plt.close()
        else:
            colors = [
                'red', 'blue', 'green', 'yellow', 'orange', 'tan', 'orangered', 'red', 'coral', 'aqua',
                'pink', 'dodgerblue', 'olive', 'aquamarine', 'azure', 'beige', 'bisque', 'violet', 'blueviolet','brown',
                'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
                'darkgoldenrod', 'dimgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon'
                ]
            for i in range(len(x)):
                #print(len(x)) #len(x) = 1
                #print(i) #0
        '''
        ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
        #save_name = name + '-{}-{}.png'.format(i, color[i])
        # txtname = self.name.split('.')[-2]  # 获取倒数第个目录名  
        # save_name = name + '_{}.png'.format(txtname)
        # save_name = os.path.join(file_dir,save_name) # save_name为点云图可视化保存路径
        #ax.scatter(x[i], y[i], z[i],s=0.1, c=colors[color[i]])
        #ax.scatter(x[i], y[i], z[i],s=0.1, c='blue')
        ax.scatter(x, y, z,s=0.1, c='blue')
        ax.axis('auto') 
        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        #ax.axis('off')  # 设置坐标轴不可见
        #ax.grid(False)  # 设置背景网格不可见
        save_path = 'visualizer/pred.png'
        plt.savefig(save_path,dpi=500) # 可调节
        plt.show() # 用于显示图形 
        plt.draw() # 重绘图
        plt.close()
        return save_path

class Child2Window(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('组件分割窗口')
        self.resize(900, 783)
    
        self.pre_cls = None
        
        # 窗体位置
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)

        # 创建布局 垂直方向
        container = QVBoxLayout()

        container.addLayout(self.init_header()) # 在布局中添加header button

        container.addLayout(self.init_title())  # 在布局中添加历史记录
        container.addLayout(self.init_table())  # 在布局中添加表格对象

        container.addLayout(self.init_label())  # 在布局中添加标签
        container.addLayout(self.init_show()) # 在布局中显示图像
        
        # 弹簧 在四个元素的最下面加弹簧，把元素顶上去
        #layout.addStretch()
        # 给窗体设置元素的排列方式
        self.setLayout(container)
 
    def parse_args(self):
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
        parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting') #投票数均值
        #print(parser.parse_args())
        return parser.parse_args()
    
    def init_header(self):
        # 1.创建顶部菜单布局
        header_layout = QHBoxLayout() # 创建对象
        # 1.1 创建按钮，加入header_layout
        btn_open_file = QPushButton('打开文件')
        btn_open_file.clicked.connect(self.openFile)
        header_layout.addWidget(btn_open_file)
        btn_seg = QPushButton('开始分割')
        btn_seg.clicked.connect(self.button_seg_click)
        header_layout.addWidget(btn_seg)
        btn_check = QPushButton('查看真实分割图片')
        btn_check.clicked.connect(self.button_check_click)
        header_layout.addWidget(btn_check)
        # 弹簧把元素弹到左侧
        header_layout.addStretch()
        return header_layout
    
    def init_title(self):
        label_layout = QHBoxLayout() # 创建标签对象
        label_layout.addStretch(1)
        title = QLabel('历史记录')
        label_layout.addWidget(title)
        label_layout.addStretch(1)
        return label_layout
    
    def init_table(self):
        # 3.创建中间表格布局
        table_layout = QHBoxLayout()  # 创建对象

        # 3.1 创建表格
        self.table_widget = table_widget = QTableWidget(0,2) # 默认显示0行2列
        self.table_widget.setFixedSize(900, 300) # 默认显示0行2列
        table_widget.setObjectName("tableWidget")
        item = QTableWidgetItem()
        item.setText("文件名") 
        table_widget.setHorizontalHeaderItem(0,item) # 设置横向标题名字
        table_widget.setColumnWidth(0,450)
        item = QTableWidgetItem()
        item.setText("所属类别") 
        table_widget.setHorizontalHeaderItem(1,item) # 设置横向标题名字
        table_widget.setColumnWidth(1,450)
        table_layout.addWidget(table_widget)
        return table_layout
        
    def init_label(self):
        label_layout = QHBoxLayout() # 创建标签对象
        label_layout.addStretch(1)
        label1 = QLabel('分割结果预测')
        label_layout.addWidget(label1)
        label_layout.addStretch(2)
        label2 = QLabel('真实分割情况')
        label_layout.addWidget(label2)
        label_layout.addStretch(1)
        return label_layout
    
    def init_show(self):
        label_layout = QHBoxLayout() # 创建标签对象
        self.label3 = label3 = QLabel()
        label3.setFixedSize(450, 400)
        label3.setText("")
        label_layout.addWidget(label3)
        self.label4 = label4 = QLabel()
        label4.setFixedSize(450, 400)
        label4.setText("")
        label_layout.addWidget(label4)
        return label_layout
    
    def openFile(self): 
        """
        打开文件，并把文件名显示到表格
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*);;Text Files (*.txt)", options=options)
        if fileName == '':
            QMessageBox.warning(self,"错误","您必需选择一个文件")
            self.status = NOTCHOOSED
            #print("# ",self.status)
            return
        self.status = CHOOSED
        self.fileName = fileName
        #获取文件名，显示到表格
        name = os.path.basename(fileName)

        # cell = QTableWidgetItem(str(name)) # 获取文件名
        # self.table_widget.setItem(0,0,cell) # 行，列放cell #传递参数，只要在定义她的时候加self line：50

        cell = QTableWidgetItem(str(name)) # 获取文件名
        self.current_row_count = current_row_count = self.table_widget.rowCount() # 获取当前表格有多少行
        self.table_widget.insertRow(current_row_count)
        self.table_widget.setItem(current_row_count,0,cell) # 行，列放cell #传递参数，只要在定义她的时候加self line：50
        return fileName 
    
    def button_seg_click(self):
        """
        分割点云图，并把分割结果显示到标签4
        """
        if not self.status:
            QMessageBox.warning(self,"错误","您必需选择一个文件")
            return

        num_classes = 50 # 填写数据集的类别数 shapenet这里就填50
        img_root = self.fileName 
        target_root = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result' # 输出结果路径
        
        #TEST_DATASET 一个txt点云文件的 point_set,cls

        TEST_DATASET = PartNormalDataset(img_root, target_root, self.pre_cls, npoints=2500, normal_channel=True)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0,drop_last=True)
        Generate_txt_and_3d_img(img_root,target_root, num_classes, testDataLoader, self.load_models_for_seg(),visualize = True)
        
        folder_name = os.path.dirname(img_root).split('/')[-1]  # 获取倒数第一个目录名 “02691156”
        self.cat = {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627',
        'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806',
        'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987',
        'Table': '04379243'}
        #通过文件名在字典中找到对应的类名
        # 遍历字典的键  
        for key, value in self.cat.items():
            # 检查当前键的值是否是我们要找的  
            if value == folder_name:  
                # 如果是，将键添加到列表中  
                lab = key  
        # 将标签显示在表格上
        cell = QTableWidgetItem(str(lab))
        current_row_count = self.current_row_count
        self.table_widget.setItem(current_row_count,1,cell) # 行，列 放cell
        self.current_row_count = current_row_count + 1

        # predict_pcd存放的是分割后图像路径 
        predict_pcd = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result/_predict_image/0_None_img.png'
        # 将save_path中的路径图片显示到标签
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        img = QPixmap(predict_pcd).scaled(self.label3.width(), self.label3.height())

        # 在label控件上显示选择的图片
        self.label3.setPixmap(img)
        self.label3.setScaledContents(True)

    def button_check_click(self):
        #label_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result/label_txt/0_label.txt' 
        lab_img_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result/label_3d_img/0_None_img.png'
        #lab_img_path = self.draw_each_img(label_path, 0, save_path = lab_img_path, heat_maps = False)
        # 将save_path中的路径图片显示到标签
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        img = QPixmap(lab_img_path).scaled(self.label4.width(), self.label4.height())
        #将参数显示在文本上
        #self.text_edit.setText(str(args))
        # 在label控件上显示选择的图片
        self.label4.setPixmap(img)
        self.label4.setScaledContents(True)

    def load_models_for_seg(self, model_dict={'PonintNet': [pointnet2(num_classes=50,normal_channel=True).eval(),
                                          r'./log/part_seg/pointnet2_part_seg_msg/checkpoints']}):
        model = list(model_dict.values())[0][0]
        checkpoints_dir = list(model_dict.values())[0][1]
        device = torch.device('cpu')
        weight_dict = torch.load(os.path.join(checkpoints_dir,'best_model.pth'),  map_location = device)
        model.load_state_dict(weight_dict['model_state_dict'])
        return model

class PartNormalDataset(Dataset):
    def __init__(self, img_root, target_root, pre_cls, npoints=2500, normal_channel=False):
        self.npoints = npoints # 采样点数
        self.img_root = img_root
        self.target_root = target_root # seg label save path
        self.pre_cls = pre_cls
        self.cat = {}
        self.normal_channel = normal_channel # 是否使用法向信息
        
        self.cat = {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627',
        'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806',
        'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987',
        'Table': '04379243'}
        
        self.classes = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8,
        'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

    def __getitem__(self, index):
        '''
        cat = list(self.cat.keys())[0] # list[0] = 'Airplane'
        fn = self.datapath[index] # 根据索引 拿到训练数据的路径self.datepath是一个元组（类名，路径）#[(Airplane,路径1),(Airplane,路径2),..]
        cat = self.datapath[index][0] # 拿到类名
        '''
        #modelnet40 ',' 分类 shapenet ' ' 分割
        self.dataset_name = dataset_name = os.path.dirname(self.img_root).split('/')[-2]  # 获取倒数第二个目录名  
        if dataset_name == 'shapenetcore_partanno_segmentation_benchmark_v0_normal':
            d = ' ' 
            # 从文件名拿到类名id
            folder_name = os.path.dirname(self.img_root).split('/')[-1]  # 获取倒数第一个目录名 “02691156”
            #通过文件名在字典中找到对应的类名
            # 遍历字典的键  
            for key, value in self.cat.items():
                # 检查当前键的值是否是我们要找的  
                if value == folder_name:  
                    # 如果是，将键添加到列表中  
                    cat = key  
        else:
            d = ','
            # 从分类结果拿到类名cat # 40和16个类中共有的几个类名
            cat = self.pre_cls 

        cls = self.classes[cat] # 将类名转换为索引0
        cls = np.array([cls]).astype(np.int32) # [0]
        data = np.loadtxt(self.img_root, delimiter = d).astype(np.float32)
        if not self.normal_channel:  # 判断是否使用法向信息
            self.point_set = data[:, 0:3]
        else:
            self.point_set = data[:, 0:6]
        
        self.point_set[:, 0:3] = pc_normalize(self.point_set[:, 0:3]) # 做一个归一化
        choice = np.random.choice(self.point_set.shape[0], self.npoints, replace=True) # 对一个类别中的数据进行随机采样 返回索引，允许重复采样
        #choice = np.random.choice(len(seg), self.npoints, replace=True) # 对一个类别中的数据进行随机采样 返回索引，允许重复采样
        # resample
        self.point_set =  self.point_set[choice, :] # 根据索引采样
        if dataset_name == 'shapenetcore_partanno_segmentation_benchmark_v0_normal':
            seg = data[:, -1].astype(np.int32)# 拿到小类别的标签[0 0 0 ... 0 0 0]全部点的类别信息
            seg = seg[choice]
            return self.point_set, cls, seg
        else:
            return self.point_set, cls#, seg # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别

    def __len__(self):
        return 1 # batch size

class Generate_txt_and_3d_img:
    def __init__(self,img_root,target_root,num_classes,testDataLoader,model,visualize = False, color_map=None):
        self.img_root = img_root # 点云数据路径
        self.target_root = target_root  # 生成txt标签和预测结果路径
        self.testDataLoader = testDataLoader
        self.num_classes = num_classes
        self.heat_map = False # 控制是否输出heatmap
        self.visualize = visualize # 是否open3d可视化
        self.color_map = color_map
        self.dataset_name = os.path.dirname(img_root).split('/')[-2]  # 获取倒数第二个目录名  

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

    def generate_predict(self):
        if self.dataset_name == 'shapenetcore_partanno_segmentation_benchmark_v0_normal':
            for batch_id, (points, label, target) in tqdm.tqdm(enumerate(self.testDataLoader),
                                                                      total=len(self.testDataLoader),smoothing=0.9):
                #点云数据、整个图像的标签、每个点的标签、 没有归一化的点云数据（带标签）torch.Size([1, 7, 2048])
                # print("points.shape: ", points.shape) # point.shape: torch.Size([1, 2048, 6])
                points = points.transpose(2, 1)
                # torch.Size([1, 6, 2048]) 有一个txt文件，每个文件中有2048个点，每个点包含六个维度的信息
                #print("points.shape: ", points.shape) # point.shape: torch.Size([1, 6, 2048])
                xyz_feature_point = points[:, :6, :]
                # 将标签保存为txt文件 label.txt 真实值非预测
                point_set_without_normal = np.asarray(torch.cat([points.permute(0, 2, 1),target[:,:,None]],dim=-1)).squeeze(0)  # 代标签 没有归一化的点云数据  的numpy形式
                np.savetxt(os.path.join(self.label_path_txt,f'{batch_id}_label.txt'), point_set_without_normal, fmt='%.04f') # 将其存储为txt文件

                '''MODEL'''
                seg_pred, _ = self.model(points, self.to_categorical(label, 16))# label:16个类别的id  seg_pred 0,1,0,2,3
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
        else:
            for batch_id, (points, label) in tqdm.tqdm(enumerate(self.testDataLoader),
                                                                      total=len(self.testDataLoader),smoothing=0.9):
                #点云数据、整个图像的标签、每个点的标签、 没有归一化的点云数据（带标签）torch.Size([1, 7, 2048])
                # print("points.shape: ", points.shape) # point.shape: torch.Size([1, 2048, 6])
                points = points.transpose(2, 1)
                # torch.Size([1, 6, 2048]) 有一个txt文件，每个文件中有2048个点，每个点包含六个维度的信息
                #print("points.shape: ", points.shape) # point.shape: torch.Size([1, 6, 2048])
                xyz_feature_point = points[:, :6, :]
                # 将标签保存为txt文件 label.txt 真实值非预测
                #point_set_without_normal = np.asarray(torch.cat([points.permute(0, 2, 1),target[:,:,None]],dim=-1)).squeeze(0)  # 代标签 没有归一化的点云数据  的numpy形式
                #np.savetxt(os.path.join(self.label_path_txt,f'{batch_id}_label.txt'), point_set_without_normal, fmt='%.04f') # 将其存储为txt文件

                '''MODEL'''
                seg_pred, _ = self.model(points, self.to_categorical(label, 16))# label:16个类别的id  seg_pred 0,1,0,2,3
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
        '''调用matpltlib 画3d图像'''
        #os.listdir用于列出指定目录下的所有文件和子目录的名称。

        each_label = os.listdir(self.label_path_txt)  # 拿到所有标签txt路径
        lab_txt_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result/label_txt'
        lab_img_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result/label_3d_img'
    
        each_pre_path = os.listdir(self.all_pred_txt_path) # 拿到所有预测后txt文件的路径
        pre_txt_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result/_predict_txt'
        pre_img_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/result/_predict_image'
        for idx,(pre_txt, lab_txt) in tqdm.tqdm(enumerate(zip(each_pre_path, each_label)),total=len(each_pre_path)):
            #self.draw_each_img(os.path.join(self.label_path_txt, lab), idx,heat_maps=False)
            # 可视化label_txt
            # 选择的是shapenet数据集
            if self.dataset_name == 'shapenetcore_partanno_segmentation_benchmark_v0_normal':
                self.draw_each_img(os.path.join(lab_txt_path, lab_txt),idx, save_path = lab_img_path, heat_maps = self.heat_map)
            # 可视化pre_txt
            self.draw_each_img(os.path.join(pre_txt_path, pre_txt),idx, save_path = pre_img_path, heat_maps = self.heat_map)
            
        print(f'所有预测图片已生成完毕，请前往：{self.all_pred_image_path} 查看')

    def draw_each_img(self, root, idx, name = None, skip = 1, save_path = None, heat_maps = False):
        "root：每个txt文件的路径"
        points = np.loadtxt(root)[:, :3]  # 点云的xyz坐标
        points_all = np.loadtxt(root)  # 点云的所有坐标
        points = pc_normalize(points)
        skip = skip  # Skip every n points

        # 创建一个新的图形窗口 
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        ax = plt.subplot(projection='3d')

        point_range = range(0, points.shape[0], skip)  # skip points to prevent crash
        #print("point_range: ", point_range) #range(0, 2048)
        x = points[point_range, 0]
        z = points[point_range, 1]
        y = points[point_range, 2]

        "根据传入的类别数 自定义生成染色板  标签 0对应 随机颜色1  标签1 对应随机颜色2"
        if self.color_map is not None:
            color_map = self.color_map
        else:
            color_map  = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, 50))} #50
        Label = points_all[point_range, -1] # 拿到标签 取(0~2048)行,-1表示选择每行的最后一个元素
        print("label: ", Label)
        # 将标签传入前面的字典，找到对应的颜色 并放入列表
        Color = list(map(lambda x: color_map[x], Label))
        #每个点的部件类别（索引0~49 mistake出现了-1）对应一个颜色
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
        #ax.axis('off')  # 设置坐标轴不可见
        #ax.grid(False)  # 设置背景网格不可见
        #ax.view_init(elev=0, azim=0) 
        #设置了子图的视角。elev=0表示观察者的仰角为0度，即观察者位于x-y平面的上方。azim=0表示观察者的方位角为0度，即观察者正面对着x轴的正方向。

        if save_path is None:
            plt.savefig(os.path.join(self.label_path_3d_img,f'{idx}_label_img.png'), dpi=300, bbox_inches='tight',transparent=True)
        else:
            plt.savefig(os.path.join(save_path, f'{idx}_{name}_img.png'), dpi=300, bbox_inches='tight',
                        transparent=True)
        plt.close()
        save_img = os.path.join(self.label_path_3d_img,f'{idx}_label_img.png')
        return save_img

    def make_dir(self, root):
        if os.path.exists(root):
            print(f'{root} 路径已存在 无需创建')
        else:
            os.mkdir(root)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
        if (y.is_cuda):
            return new_y.cuda()
        return new_y
 
class FatherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('三维点云图下的物体识别于组件分割系统')
        self.resize(400, 300)

        btn1 = QPushButton('物体识别', self)
        #btn1.setFixedSize(150,30)
        btn1.clicked.connect(self.btn1Clicked)

        btn2 = QPushButton('组件分割', self)
        #btn2.setFixedSize(150,30)
        btn2.clicked.connect(self.btn2Clicked)

        layout = QVBoxLayout()
        layout.addStretch(2)
        layout.addWidget(btn1)
        layout.addStretch(1)
        layout.addWidget(btn2)
        layout.addStretch(2)
        self.setLayout(layout)
 
        self.show()
 
    def btn1Clicked(self):
        self.chile_Win = Child1Window()
        self.chile_Win.show()
        self.chile_Win.exec_()

    def btn2Clicked(self):
        self.chile_Win = Child2Window()
        self.chile_Win.show()
        self.chile_Win.exec_() 
 
if __name__ == '__main__':
 
    app = QApplication(sys.argv)
    # 创建主窗口
    window = FatherWindow()
    # 显示窗口
    window.show()
    # 运行应用，并监听事件
    sys.exit(app.exec_())