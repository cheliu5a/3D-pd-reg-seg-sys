import os
import sys
import importlib
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QRect

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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 窗体标题和尺寸
        self.setWindowTitle('三维点云图下的物体识别与组件分割系统')
        # 窗体的尺寸
        self.resize(994, 783)
        
        # 窗体位置
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)

        # 创建布局 垂直方向
        layout = QVBoxLayout()

        layout.addLayout(self.init_header()) # 在布局中添加header
        layout.addLayout(self.init_table())  # 在布局中添加表格对象
        layout.addLayout(self.init_label())  # 在布局中添加标签
        layout.addLayout(self.init_show()) # 在布局中显示图像
        
        # 弹簧 在四个元素的最下面加弹簧，把元素顶上去
        #layout.addStretch()
        # 给窗体设置元素的排列方式
        self.setLayout(layout)

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
        parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
        print(parser.parse_args())
        return parser.parse_args()

    def init_header(self):
        # 1.创建顶部菜单布局
        header_layout = QHBoxLayout() # 创建对象
        # 1.1 创建按钮，加入header_layout
        btn_open_file = QPushButton('打开文件')
        btn_open_file.clicked.connect(self.openFile)
        header_layout.addWidget(btn_open_file)
        btn_cls = QPushButton('物体识别')
        btn_cls.clicked.connect(self.button_cls_click)
        header_layout.addWidget(btn_cls)
        btn_seg = QPushButton('组件分割')
        header_layout.addWidget(btn_seg)
        # 弹簧把元素弹到左侧
        #header_layout.addStretch()
        return header_layout

    def init_table(self):
        # 3.创建中间表格布局
        table_layout = QHBoxLayout()  # 创建对象

        # 3.1 创建表格
        self.table_widget = table_widget = QTableWidget(0,2) # 默认显示0行2列
        self.table_widget.setFixedSize(994, 300) # 默认显示0行2列
        table_widget.setObjectName("tableWidget")
        item = QTableWidgetItem()
        item.setText("文件名") 
        table_widget.setHorizontalHeaderItem(0,item) # 设置横向标题名字
        table_widget.setColumnWidth(0,500)
        item = QTableWidgetItem()
        item.setText("识别结果") 
        table_widget.setHorizontalHeaderItem(1,item) # 设置横向标题名字
        table_widget.setColumnWidth(1,500)
        table_layout.addWidget(table_widget)
        return table_layout
        
    def init_label(self):
        label_layout = QHBoxLayout() # 创建标签对象
        label_layout.addStretch(1)
        label1 = QLabel('三维点云图')
        label_layout.addWidget(label1)
        label_layout.addStretch(2)
        label2 = QLabel('组件分割图')
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
        self.fileName = fileName
        #print("文件已打开")
        #获取文件名，显示到表格
        #print(fileName)
        name = os.path.basename(fileName)
        #print(name)

        # cell = QTableWidgetItem(str(name)) # 获取文件名
        # self.table_widget.setItem(0,0,cell) # 行，列放cell #传递参数，只要在定义她的时候加self line：50

        cell = QTableWidgetItem(str(name)) # 获取文件名
        self.current_row_count = current_row_count = self.table_widget.rowCount() # 获取当前表格有多少行
        self.table_widget.insertRow(current_row_count)
        #print("当前表格第:",current_row_count)
        self.table_widget.setItem(current_row_count,0,cell) # 行，列放cell #传递参数，只要在定义她的时候加self line：50
        #print("当前表格第:",current_row_count)
        return fileName
    
    def button_cls_click(self):
        """
        识别文件，并把识别结果显示到表格
        """

        dataset=self.fileName
        # print(dataset)
        # print(type(dataset))
        pcdataset = np.loadtxt(dataset, delimiter=',').astype(np.float32)#数据读取，我的数据是三（6）个维度，数据之间是空格，如果是逗号修改一下即可
        
        point_set = pcdataset[0:10000, :] #我的输入数据设置为原始数据中10000个点
        point_set[:, 0:6] = self.pc_normalize(point_set[:, 0:6]) #归一化数据
        point_set = point_set[:, 0:6] 
        point_set = point_set.transpose(1,0)#将数据由N*C转换为C*N
        point_set = point_set.reshape(1, 6, 10000)
        self.n_points = point_set # 赋值给这个类的对象，通过self.n_points传值
        point_set = torch.as_tensor(point_set) # 需要将数据格式变为张量，不然会报错

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
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location = device) 
        classifier.load_state_dict(checkpoint['model_state_dict'])
        #预测分类
        with torch.no_grad():
            pred_label, img_path = self.test(classifier.eval(), point_set, vote_num=args.num_votes, num_class=num_class)

        # 将标签显示在表格上
        result = pred_label
        cell = QTableWidgetItem(str(result))
        current_row_count = self.current_row_count
        #current_row_count = self.table_widget.rowCount() # 获取当前表格有多少行 会使得currow加1
        self.table_widget.setItem(current_row_count,1,cell) # 行，列放cell
        self.current_row_count = current_row_count + 1

        # 将save_path中的路径图片显示到标签
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        img = QPixmap(img_path).scaled(self.label3.width(), self.label3.height())

        # 在label控件上显示选择的图片
        self.label3.setPixmap(img)
        self.label3.setScaledContents(True)
        
    def pc_normalize(self,pc):  #点云数据归一化
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    #分类测试函数
    def test(self, model, point_set, num_class=40, vote_num=1):
        # 返回标签和图片保存路径
        classifier = model.eval()
        #class_acc = np.zeros((num_class, 3))
        vote_pool = torch.zeros(1, num_class)

        for _ in range(vote_num):
            pred, _ = classifier(point_set)
            #print(pred) # 3次 num_votes=3
            vote_pool += pred
        pred = vote_pool / vote_num
        #print(pred)
        # 对预测结果每行取最大值得到分类
        pred_choice = pred.data.max(1)[1]
        #print(pred_choice) # tensor([num])
        #print(pred_choice.item()) # num 
        #print(LABEL_MAPPING[pred_choice.item()])
        
        #可视化
        file_dir = '/home/chen/Pointnet_Pointnet2_pytorch-master/visualizer'
        save_name_prefix = 'pred'
        n_points = self.n_points # 通过self.n_points传值
        save_path = self.draw(n_points[:, 0, :], n_points[:, 1, :], n_points[:, 2, :], save_name_prefix, file_dir, color=pred_choice)

        return LABEL_MAPPING[pred_choice.item()], save_path

    #定义可视化函数
    def draw(self, x, y, z, name, file_dir, color=None): # name:pred
        """
        绘制单个样本的三维点图,返回点云图保存路径
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
                plt.draw()
                plt.savefig(save_name)
                # plt.show()
                plt.close()
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
                save_name = save_name = os.path.join(file_dir,save_name) # save_name为点云图可视化保存路径
                ax.scatter(x[i], y[i], z[i],s=0.1, c=colors[color[i]])
                ax.set_zlabel('Z')  # 坐标轴
                ax.set_ylabel('Y')
                ax.set_xlabel('X')
                plt.draw()
                plt.savefig(save_name) 
                # plt.show()
                plt.close()
        return save_name

if __name__ == '__main__':
    app = QApplication(sys.argv) # 创建application的对象
    window = MainWindow() # 调用类，实例化，执行init方法
    window.show()
    sys.exit(app.exec_())


