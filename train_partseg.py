import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
'''
映射关系改变
'''
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

'''
将神经网络模型中的 ReLU 激活函数设置为原地（inplace）操作
原地操作表示将会进行原地操作（即直接在原有的内存空间上修改数据），从而减少内存的使用，提高模型的运行效率。
'''
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
'''
将整数标签转换为独热编码（one-hot encoding） 
'''
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

'''
命令行参数设置，也是整个网络运行的基本设置，具体含义见后。 与分类代码基本一样
--model：模型名称，类型为字符串，默认值为 'pointnet_part_seg'，用于指定使用哪个模型进行训练。
--batch_size：批次大小，类型为整数，默认值为 16，表示在训练过程中每次传入模型的数据样本数。
--epoch：训练轮数，类型为整数，默认值为 251，表示训练过程中总共需要迭代多少次。
--learning_rate：初始学习率，类型为浮点数，默认值为 0.001，表示在训练过程中初始的学习率。
--gpu：指定 GPU 设备，类型为字符串，默认值为 '0'，表示在训练过程中使用哪一块 GPU 设备进行计算，可以同时指定多块 GPU 设备，例如 '0,1,2'。
--optimizer：优化器类型，类型为字符串，默认值为 'Adam'，表示在训练过程中使用哪种优化器进行模型参数的更新，可以选择 'Adam' 或 'SGD'。
--log_dir：日志路径，类型为字符串，默认值为 None，表示训练过程中保存日志文件的路径。
--decay_rate：权重衰减，类型为浮点数，默认值为 1e-4，表示在训练过程中的权重衰减率。
--npoint：点云采样数，类型为整数，默认值为 2048，表示在训练过程中每个点云模型需要采样多少个点。
--normal：是否使用法向量，类型为布尔值，默认值为 False，表示在训练过程中是否使用点云模型的法向量信息。
--step_size：学习率衰减步长，类型为整数，默认值为 20，表示在训练过程中学习率需要下降的迭代步数。
--lr_decay：学习率衰减倍数，类型为浮点数，默认值为 0.5，表示在每个学习率衰减步长结束后学习率需要下降的倍数。
'''
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    return parser.parse_args()


def main(args):
    '''
    main(args)一开始与分类训练代码中一样，都是一些基本参数的设置，训练信息保存的路径，训练数据的加载，模型读取。 
    '''
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
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

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    '''
    这段代码定义了一个名为 weights_init 的函数，用于对模型参数进行初始化。具体来说，当模型的某个子模块是 Conv2d 或 Linear 类型时，
    将其权重矩阵使用 Xavier 初始化方法进行初始化，将其偏置向量全部初始化为 0。
    而Xavier 初始化方法是一种常用的权重初始化方法，它可以使模型在训练过程中更快地收敛，并提高模型的泛化性能。
    在这里，使用该函数对模型参数进行初始化，可以提高模型的训练效果。
    '''
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    '''
    查看是否有预训练模型，然后设置优化器参数。
    '''
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    '''
    这段代码定义了一个名为 bn_momentum_adjust 的函数，用于调整 Batch Normalization 层的动量参数。
    具体来说，当模型的某个子模块是 BatchNorm2d 或 BatchNorm1d 类型时，将其动量参数设置为函数输入的 momentum 值。
    Batch Normalization 层是一种常用的神经网络层，它可以加速模型的训练过程，并提高模型的泛化性能。
    其中，动量参数 momentum 用于平滑 Batch Normalization 层中均值和方差的计算过程。
    在这里，使用该函数对模型中的 Batch Normalization 层的动量参数进行设置，可以进一步优化模型的训练效果。

    '''
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
    '''
    学习率裁剪的阈值 LEARNING_RATE_CLIP，原始动量值 MOMENTUM_ORIGINAL，动量衰减的比例 MOMENTUM_DECCAY，
    以及动量衰减的步数 MOMENTUM_DECCAY_STEP。
    具体来说，LEARNING_RATE_CLIP 用于限制学习率的最小值，避免学习率过小导致收敛缓慢。MOMENTUM_ORIGINAL 表示模型中 
    Batch Normalization 层的原始动量值，MOMENTUM_DECCAY 表示动量衰减的比例，用于控制动量在每个训练阶段的变化程度。
    MOMENTUM_DECCAY_STEP 则表示动量衰减的步数，即每经过 MOMENTUM_DECCAY_STEP 个训练 epoch，动量的值就乘上 MOMENTUM_DECCAY。
    '''
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    
    # 最佳准确率 best_acc，全局训练轮次 global_epoch，
    # 最佳类别平均交并比 best_class_avg_iou，以及最佳实例平均交并比 best_instance_avg_iou
    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            # 优化器梯度清零
            optimizer.zero_grad()

            # 数据增强
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # 数据被转换为PyTorch的Tensor格式，并移动到GPU上加速计算。
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            
            #表示对一个batch的数据(points)进行前向传播计算，生成模型的预测结果(seg_pred)和转换特征(trans_feat)。
            #其中，to_categorical(label, num_classes)将标签(label)转换为一个one-hot向量，以便在模型中使用。
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            #将预测结果(seg_pred)进行形状变换，将其变为一个2D矩阵，其中每行表示一个数据点的预测结果，每列表示一个类别的概率值。
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            #将目标(target)进行形状变换，将其变为一个1D张量，其中每个元素表示一个数据点的目标类别。
            target = target.view(-1, 1)[:, 0]
            # 将预测结果(seg_pred)中每个数据点概率最大的类别作为预测结果(pred_choice)，并返回一个1D张量。
            pred_choice = seg_pred.data.max(1)[1] 

            # 计算该batch中（24个）正确分类的点的数量
            correct = pred_choice.eq(target.data).cpu().sum() #计算预测结果(pred_choice)与目标(target)相等的数据点个数(correct)。
            # 计算当前batch的预测准确率(mean_correct)。（args.batch_size和args.npoint分别表示batch的大小和每个数据点的最大数量。）
            mean_correct.append(correct.item() / (args.batch_size * args.npoint)) 

            #计算损失并反向传播：使用损失函数计算预测结果与真实标签之间的损失，并进行反向传播更新模型的权重和偏置。
            # 计算当前batch的损失值(loss)，其中criterion是一个损失函数（seg_pred和target分别表示模型的预测结果和目标，trans_feat表示转换特征。）
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward() # 进行反向传播算法，计算模型参数的梯度。
            optimizer.step() # 根据计算的梯度更新模型的权重和偏置。
        
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        #训练完一轮后就要进行测试集的评估
        with torch.no_grad():#设置上下文环境，禁用梯度计算，以加快推理速度。
            test_metrics = {}#test_metrics = {}用于定义一个字典，用于保存测试集的各项指标。
            #分别用于统计测试集中所有数据点的预测正确数和总数。
            total_correct = 0
            total_seen = 0
            #分别用于统计测试集中各个类别的数据点总数和预测正确数，其中num_part表示数据点的类别数。
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            #用于定义一个字典，用于保存每个类别的平均交并比(IOU)。
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            #用于定义一个字典，用于将数据点的标签映射到类别名称，其中seg_classes是一个保存类别名称的字典。
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()
            
            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                '''
                当前测试批次
                points:torch.Size([12, 1024, 6]) [B,N,D]
                target:torch.Size([12]) 十二张图像的标签
                '''
                #数据加载与预处理
                cur_batch_size, NUM_POINT, _ = points.size()#获取当前批次和点云数据，其中cur_batch_size是批次大小，NUM_POINT是每个样本中的点数。
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                # 模型前向传播
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                # 获取预测值并处理
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                #初始化一个全零数组cur_pred_val，用于存储最终的预测类别。
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()
                # 计算16个样本（当前batch）中每个点的预测类别：
                for i in range(cur_batch_size):#用于对当前batch中的每个数据点进行处理。
                    cat = seg_label_to_cat[target[i, 0]]#获取当前数据点的类别名称。
                    logits = cur_pred_val_logits[i, :, :]# 获取当前数据点的预测结果的logits。
                    '''
                    将logits转换为每个点的类别标签，并将其保存到cur_pred_val数组中。
                    其中，np.argmax()函数用于获取logits中最大值的索引，即预测类别的编号；
                    seg_classes[cat]用于获取当前类别所对应的类别编号列表，再加上[0]是因为我们通常只需要其中任意一个编号即可。
                    最后，将预测类别编号加上当前类别的第一个编号，即可得到当前点的类别标签。
                    '''
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]#对每个点，
                    
                # 计算当前批次中预测正确的点数
                '''
                计算当前batch的预测结果中有多少与目标值相同。其中，cur_pred_val是模型的预测结果，target是目标值，它们都是numpy数组。
                在这里，cur_pred_val == target会生成一个布尔值数组，其中每个元素表示对应位置的预测值是否等于目标值。
                np.sum()函数用于计算所有True的数量，即当前batch中预测正确的点的数量。
                '''
                correct = np.sum(cur_pred_val == target)
                # 更新总正确数total_correct和总点数total_seen。
                total_correct += correct#将当前batch中预测正确的点的数量累加到总的正确预测数中。
                total_seen += (cur_batch_size * NUM_POINT)#统计当前batch中一共有多少个点，其中cur_batch_size是当前batch的大小，NUM_POINT是每个点云中的最大点数。
                # 遍历每个类别，累计每个类别的观察点数和正确点数。
                '''
                这段代码用于统计当前batch中每个类别的预测结果中有多少与目标值相同，并将其累加到每个类别的总的正确预测数中。
                同时，也统计了当前batch中每个类别一共有多少个点。
                '''
                for l in range(num_part):#对每个类别进行处理。其中，num_part是数据集中类别的数量。
                    #统计当前batch中每个类别一共有多少个点，其中target == l会生成一个布尔值数组，其中每个元素表示对应位置的目标值是否属于第l类。
                    #np.sum()函数用于计算所有True的数量，即当前batch中第l类的点的数量。
                    total_seen_class[l] += np.sum(target == l)
                    '''
                    统计当前batch中每个类别的预测结果中有多少与目标值相同，其中(cur_pred_val == l) & (target == l)会生成一个布尔值数组，其中每个元素表示对应位置的预测值和目标值是否都属于第l类。
                    np.sum()函数用于计算所有True的数量，即当前batch中第l类预测正确的点的数量。
                    '''
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))
                # 计算每个形状的每个部分的IoU,并将其保存在一个列表中，最终计算每个形状的平均IoU。
                for i in range(cur_batch_size):#对当前batch中的每个点云进行处理。
                    segp = cur_pred_val[i, :]#获取当前点云的预测结果。
                    segl = target[i, :]#获取当前点云的目标值。
                    cat = seg_label_to_cat[segl[0]]#获取当前点云所属的形状的类别。
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]#初始化一个长度为当前形状部分数量的全零列表，用于保存每个部分的IoU。
                    for l in seg_classes[cat]:#对当前形状的每个部分进行处理。其中，seg_classes是一个字典，用于保存每个类别的部分标签。
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # 判断当前部分是否在目标值和预测结果中都不存在，如果是，则将该部分的IoU设为1.0。
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        #计算当前部分的IoU，具体来说，它先计算当前部分在目标值和预测结果中同时存在的点的数量，
                        #然后除以当前部分在目标值和预测结果中存在的点的数量的并集。
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    #将当前形状的平均IoU添加到一个列表中，该列表用于保存每个形状的平均IoU。 
                    shape_ious[cat].append(np.mean(part_ious))
            '''所有测试批次一个测试集'''
            all_shape_ious = []
            #计算每个类别的平均IoU和收集所有IoU 这段代码遍历每个类别，并计算该类别的所有IoU值的平均值。同时，所有的IoU值都被收集到all_shape_ious列表中，以便后续计算实例级别的平均IoU。
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            # 计算类别平均IoU
            mean_shape_ious = np.mean(list(shape_ious.values()))
            # 计算准确率和其他评估指标:
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        # 保存最佳模型
        #把当前在测试集上的结果与以前的进行比较，保存最好的那个。以及将一些结果打印出来并写入训练日志。
        #如果当前计算的实例平均IoU大于或等于之前保存的最佳实例平均IoU，则保存当前模型的状态到best_model.pth文件中。
        #这里假设checkpoints_dir是一个目录路径，而state是一个包含模型参数和其他可能需要的信息的字典。
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        # 更新最佳评估指标
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
