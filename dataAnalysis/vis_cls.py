import re  
import matplotlib.pyplot as plt  
  
# 日志文件路径  
log_file_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/log/classification/pointnet2_msg_normals/logs/pointnet2_cls_msg.txt'  # 请替换为实际的日志文件路径  

# 用于存储提取的数据  
epochs = []  
train_instance_accuracies = []
test_instance_accuracies = []  
class_accuracies = []  
best_instance_accuracies = []  
best_class_accuracies = []  
'''
2021-03-26 21:02:03,746 - Model - INFO - Epoch 3 (3/200):
2021-03-26 21:05:15,349 - Model - INFO - Train Instance Accuracy: 0.781606
2021-03-26 21:05:51,538 - Model - INFO - Test Instance Accuracy: 0.803641, Class Accuracy: 0.738575
2021-03-26 21:05:51,538 - Model - INFO - Best Instance Accuracy: 0.803641, Class Accuracy: 0.738575
2021-03-26 21:05:51,539 - Model - INFO - Save model...
2021-03-26 21:05:51,539 - Model - INFO - Saving at log/classification/pointnet2_msg_normals/checkpoints/best_model.pth
'''
# 正则表达式模式用于匹配日志中的关键信息  
patterns = {  
    'Epoch': r'Epoch (\d+) \(\d+/\d+\):',
    'train_instance_accuracy': r'Train Instance Accuracy: ([\d.]+)',  
    'test_instance_accuracy': r'Test Instance Accuracy: ([\d.]+), Class Accuracy: ([\d.]+)',  
    'best_instance_accuracy': r'Best Instance Accuracy: ([\d.]+), Class Accuracy: ([\d.]+)'  
}  
  
# 读取日志文件并提取数据  
with open(log_file_path, 'r') as file:  
    for line in file:  
        for key, pattern in patterns.items():  
            match = re.search(pattern, line)  
            if match:  
                if key == 'Epoch':  
                    epoch = int(match.group(1))
                    epochs.append(epoch)
                elif key == 'train_instance_accuracy':  
                    train_instance_accuracies.append(float(match.group(1)))  
                elif key == 'test_instance_accuracy':  
                    test_instance_accuracies.append(float(match.group(1)))  
                    class_accuracies.append(float(match.group(2)))   
                elif key == 'best_instance_accuracy':  
                    best_instance_accuracies.append(float(match.group(1)))  
                    best_class_accuracies.append(float(match.group(2)))  
                

# print(len(epochs),
# len(train_instance_accuracies),
# len(test_instance_accuracies), 
# len(class_accuracies),
# len(best_instance_accuracies),
# len(best_class_accuracies))

# 绘制折线图  
'''
marker：数据点的标记样式。例如，'^'表示三角形，'o'表示圆圈，'s'表示正方形，'D'表示菱形。
linestyle：线的样式。例如，'--'表示虚线，'-'表示实线。
color：线的颜色。例如，'r'表示红色，'g'表示绿色，'b'表示蓝色。
'''
plt.figure(figsize=(12, 6))  
plt.plot(epochs, train_instance_accuracies, label='Train Instance Accuracy', marker='^', linestyle='--')  
plt.plot(epochs, test_instance_accuracies, label='Test Instance Accuracy', marker='o', linestyle='-')  
plt.plot(epochs, class_accuracies, label='Test Class Accuracy', marker='^', linestyle='--', color='r')  
plt.plot(epochs, best_instance_accuracies, label='Best Instance Accuracy', marker='^', linestyle='-')  
plt.plot(epochs, best_class_accuracies, label='Best Class Accuracy', marker='D', linestyle='-')  

# # 在best_instance_accuracies发生变化时标注出来  
# for i in range(1, len(best_instance_accuracies)):  
#     if best_instance_accuracies[i] != best_instance_accuracies[i-1]:  
#         # 创建标注文本  f'Epoch {epochs[i]}: Best Instance Accuracy = {best_instance_accuracies[i]:.6f}'
#         if epochs[i] >= 23:   
#             annotation_text = f'Epoch:{epochs[i]}'  
#             plt.annotate(annotation_text,   
#                         (epochs[i], best_instance_accuracies[i]),   
#                         textcoords="offset points",   
#                         xytext=(0,10),   
#                         ha='center')  
            
# 在每个数据点上添加注释，根据y值决定注释的位置  
# 在best_instance_accuracies发生变化时标注出来  
a = 0
for i in range(1, len(best_instance_accuracies)):  
    if best_instance_accuracies[i] != best_instance_accuracies[i-1]:
        a += 1
        # 创建标注文本  f'Epoch {epochs[i]}: Best Instance Accuracy = {best_instance_accuracies[i]:.6f}'
        if epochs[i] >= 20:   
            annotation_text = f'{epochs[i]},{best_instance_accuracies[i]:.2f}'  
            
            # 根据y值决定注释的位置  
            if a % 2 == 0:  # 如果y值大于中间值，放在上方  
                xytext = (0, 10)  
            else:  # 如果y值小于等于中间值，放在下方  
                xytext = (0, -10)  
            
            plt.annotate(annotation_text,  
                        (epochs[i], best_instance_accuracies[i]),  
                        textcoords="offset points",  
                        xytext=xytext,  
                        ha='center',  
                        va='center')  # 垂直对齐方式设置为居中  
# 添加图例  
plt.legend()  
  
# 添加标题和坐标轴标签  
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')  
plt.ylabel('Accuracy')  
  
# 设置x轴的范围，确保所有点都可见  
plt.xlim(min(epochs) - 1, max(epochs) + 1)  
  
# 设置y轴的范围以适应所有数据  
plt.ylim(min(train_instance_accuracies + test_instance_accuracies + class_accuracies) - 0.05,  
         max(train_instance_accuracies + test_instance_accuracies + class_accuracies) + 0.05)  
  
# 显示网格  
#plt.grid(True)  
  
# 显示图表  
plt.show()

# import re  
# import matplotlib.pyplot as plt  
  
# # 日志文件路径  
# log_file_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/log/classification/pointnet2_msg_normals/logs/pointnet2_cls_msg.txt'  # 请替换为你的日志文件路径  
  
# # 用于存储解析出的数据  
# data = {  
#     'epoch': [],  
#     'train_instance_accuracy': [],  
#     'test_instance_accuracy': [],  
#     'class_accuracy': [],  
#     'best_instance_accuracy': [],  
#     'best_class_accuracy': []  
# }  
  
# # 正则表达式模式用于匹配日志中的关键信息  
# patterns = {  
#     'epoch': r'Epoch (\d+) \(\d+/\d+\):',  
#     'train_instance_accuracy': r'train_instance_accuracy: ([\d.]+)',  
#     'test_instance_accuracy': r'Test Instance Accuracy: ([\d.]+), Class Accuracy: ([\d.]+)',  
#     'best_instance_accuracy': r'Best Instance Accuracy: ([\d.]+), Class Accuracy: ([\d.]+)'  
# }  
  
# # 读取日志文件并解析数据  
# with open(log_file_path, 'r') as file:  
#     for line in file:  
#         for key, pattern in patterns.items():  
#             match = re.search(pattern, line)  
#             if match:  
#                 if key == 'epoch':  
#                     data['epoch'].append(int(match.group(1)))  
#                 else:  
#                     data[key].append(float(match.group(1)))  
#                     if key == 'test_instance_accuracy':  
#                         data['class_accuracy'].append(float(match.group(2)))  
#                     elif key == 'best_instance_accuracy':  
#                         data['best_class_accuracy'].append(float(match.group(2)))  
#                 break  # 找到匹配项后跳出循环，继续处理下一行  


# ########################
# # 绘制图表  
# plt.figure(figsize=(12, 6))  
  
# # 绘制所有准确度指标  
# colors = ['blue', 'orange', 'green', 'red', 'purple']  # 为每个指标分配不同的颜色  
# labels = ['Train Instance Accuracy', 'Test Instance Accuracy', 'Test Class Accuracy', 'Best Instance Accuracy', 'Best Class Accuracy']  
  
# for i, (key, color) in enumerate(zip(list(data.keys())[1:], colors)):  # 跳过'epoch'键  
#     plt.plot(data['epoch'], data[key], color=color, label=labels[i])  
  
# # 设置图表标题和坐标轴标签  
# plt.title('Accuracy Metrics Over Epochs')  
# plt.xlabel('Epoch')  
# plt.ylabel('Accuracy')  
  
# # 显示图例  
# plt.legend()  
  
# # 显示图表  
# plt.show()
