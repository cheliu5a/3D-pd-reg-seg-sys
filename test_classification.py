from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse #python的命令行解析的模块，内置于python，不需要安装
import numpy as np
import os
import torch
import logging #日志处理
from tqdm import tqdm #进度条模块
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args(): #解析命令行参数
    '''PARAMETERS'''
    # 建立参数解析对象
    parser = argparse.ArgumentParser('Testing')
    # 添加属性：给xx实例增加一个aa属性，如xx.add_argument('aa')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root') #指定的日志目录
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals') #是否运用法向量信息
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting') #分类投票
    return parser.parse_args() #采用parser对象的parse_args函数获取解析的参数


def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval() #测试时不启用BatchNormalization和Dropout 使用train函数则启用
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda() #张量shape都是默认的batch_size,即24

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num): #default：3
            pred, _ = classifier(points) #分类器来处理数据
            vote_pool += pred
        pred = vote_pool / vote_num #求vote_num次数的平均
        pred_choice = pred.data.max(1)[1] #在所有的类别的得分中取一个最大值

        #求类别的accuracy每个类别中的正确判断数
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    #求类别的accuracy
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    #求instance的acc 对所有的不分类别的acc统计
    instance_acc = np.mean(mean_correct) #mean_correct list(103) 2468/24(batch)=103
    return instance_acc, class_acc 


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model") 
    logger.setLevel(logging.INFO) #日志级别
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    classifier = model.get_model(num_class, normal_channel=args.use_normals) #获取网络模型
    
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')#训练好的网络模型参数
    classifier.load_state_dict(checkpoint['model_state_dict']) #把网络模型加载进来，构造网络

    with torch.no_grad(): #运用classifier.eval()推理
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
