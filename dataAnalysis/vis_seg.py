import re  
import matplotlib.pyplot as plt  
  
# 日志文件路径  
log_file_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/log/part_seg/pointnet2_part_seg_msg/logs/pointnet2_part_seg_msg.txt'  # 请替换为实际的日志文件路径  
  
# 用于存储提取的数据  
epochs = []  
learning_rates = []
train_accuracies = []  
test_accuracies = []  
class_avg_mious = []  
instance_avg_mious = []  
best_accuracies = []  
best_class_avg_mious = []  
best_instance_avg_mious = []  
'''
2021-03-27 10:42:20,188 - Model - INFO - Epoch 107 (107/251):
2021-03-27 10:42:20,188 - Model - INFO - Learning rate:0.000031
2021-03-27 10:48:56,194 - Model - INFO - Train accuracy is: 0.95616
2021-03-27 10:50:00,972 - Model - INFO - Epoch 107 test Accuracy: 0.942826  Class avg mIOU: 0.823246   Inctance avg mIOU: 0.852062
2021-03-27 10:50:00,973 - Model - INFO - Best accuracy is: 0.94388
2021-03-27 10:50:00,973 - Model - INFO - Best class avg mIOU is: 0.82608
2021-03-27 10:50:00,973 - Model - INFO - Best inctance avg mIOU is: 0.85405
'''
# 正则表达式模式，用于匹配所需的数据  
patterns = {  
    'Epoch': r'Epoch (\d+) \(\d+/\d+\):',  
    'Learning rate':r'Learning rate:(\d+\.\d+)',
    'Train accuracy': r'Train accuracy is: (\d+\.\d+)',  
    'test Accuracy': r'Epoch (\d+) test Accuracy: (\d+\.\d+)  Class avg mIOU: (\d+\.\d+)   Inctance avg mIOU: (\d+\.\d+)',  
    'Best accuracy': r'Best accuracy is: (\d+\.\d+)',  
    'Best class avg mIOU': r'Best class avg mIOU is: (\d+\.\d+)',  
    'Best instance avg mIOU': r'Best inctance avg mIOU is: (\d+\.\d+)'  
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
                elif key == 'Learning rate':  
                    learning_rates.append(float(match.group(1)))  
                elif key == 'Train accuracy':  
                    train_accuracies.append(float(match.group(1)))  
                elif key == 'test Accuracy':  
                    epoch = int(match.group(1))  
                    test_accuracies.append(float(match.group(2)))  
                    class_avg_mious.append(float(match.group(3)))  
                    instance_avg_mious.append(float(match.group(4)))  
                    # 注意：这里假设每个epoch只出现一次test Accuracy行  
                # 匹配并提取Best accuracy, Best class avg mIOU, Best instance avg mIOU  
                elif key == 'Best accuracy':
                    best_accuracy = float(match.group(1))  
                    best_accuracies.append(best_accuracy)  
                    #print('best_accuracy:',float(best_accuracy))  
                elif key == 'Best class avg mIOU':  
                    best_class_avg_miou = float(match.group(1))  
                    best_class_avg_mious.append(best_class_avg_miou)  
                    #print('best_class_avg_miou:',best_class_avg_miou)
                elif key =='Best instance avg mIOU':  
                    best_instance_avg_miou = float(match.group(1))  
                    best_instance_avg_mious.append(best_instance_avg_miou)  
                    #print('best_instance_avg_miou:',best_instance_avg_miou)

# print(len(epochs),
# len(learning_rates),
# len(train_accuracies), 
# len(test_accuracies),
# len(class_avg_mious),
# len(instance_avg_mious),
# len(best_accuracies), 
# len(best_class_avg_mious),
# len(best_instance_avg_mious))

# 绘制折线图  
'''
marker：数据点的标记样式。例如，'^'表示三角形，'o'表示圆圈，'s'表示正方形，'D'表示菱形。
linestyle：线的样式。例如，'--'表示虚线，'-'表示实线。
color：线的颜色。例如，'r'表示红色，'g'表示绿色，'b'表示蓝色。
'''
plt.figure(figsize=(12, 6))  
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='^', linestyle='--')  
plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o', linestyle='-')  
plt.plot(epochs, best_accuracies, label='Best Accuracy', marker='^', linestyle='--', color='r')  
plt.plot(epochs, class_avg_mious, label='Class Avg mIOU', marker='s', linestyle='-')  
plt.plot(epochs, instance_avg_mious, label='Instance Avg mIOU', marker='D', linestyle='-')  
plt.plot(epochs, best_class_avg_mious, label='Best Class Avg mIOU', marker='s', linestyle='--', color='g')  
plt.plot(epochs, best_instance_avg_mious, label='Best Instance Avg mIOU', marker='D', linestyle='--', color='b')  

a = 0
for i in range(1, len(best_instance_avg_mious)):  
    if best_instance_avg_mious[i] != best_instance_avg_mious[i-1]:
        a += 1
        # 创建标注文本  f'Epoch {epochs[i]}: Best Instance Accuracy = {best_instance_accuracies[i]:.6f}'
        # if epochs[i] >= 20:   
        annotation_text = f'{epochs[i]},{best_instance_avg_mious[i]:.2f}'  
        
        # 根据y值决定注释的位置  
        if a % 2 == 0:  # 如果y值大于中间值，放在上方  
            xytext = (0, 10)  
        else:  # 如果y值小于等于中间值，放在下方  
            xytext = (0, -10)  
        
        plt.annotate(annotation_text,  
                    (epochs[i], best_instance_avg_mious[i]),  
                    textcoords="offset points",  
                    xytext=xytext,  
                    ha='center',  
                    va='center')  # 垂直对齐方式设置为居中  
# 添加图例  
plt.legend()  
  
# 添加标题和坐标轴标签  
plt.title('Epoch vs Accuracy and mIOU Metrics')
plt.xlabel('Epoch')  
plt.ylabel('Accuracy and mIOU')  
  
# 设置x轴的范围，确保所有点都可见  
plt.xlim(min(epochs) - 1, max(epochs) + 1)  
  
# 设置y轴的范围以适应所有数据  
plt.ylim(min(train_accuracies + test_accuracies + class_avg_mious + instance_avg_mious) - 0.05,  
         max(train_accuracies + test_accuracies + class_avg_mious + instance_avg_mious) + 0.05)  
  
# 显示网格  
#plt.grid(True)  
  
# 显示图表  
plt.show()
########################################
plt.figure(figsize=(12, 6))
plt.plot(epochs, class_avg_mious, label='Class Avg mIOU', marker='s', linestyle='-')  
plt.plot(epochs, instance_avg_mious, label='Instance Avg mIOU', marker='D', linestyle='-')
plt.plot(epochs, best_class_avg_mious, label='Best Class Avg mIOU', marker='s', linestyle='--', color='g')  
plt.plot(epochs, best_instance_avg_mious, label='Best Instance Avg mIOU', marker='D', linestyle='--', color='b')  
  
# 添加图例  
plt.legend()  
  
# 添加标题和坐标轴标签  
plt.title('Epoch vs mIOU Metrics')
plt.xlabel('Epoch')  
plt.ylabel('mIOU')  
  
# 设置x轴的范围，确保所有点都可见  
plt.xlim(min(epochs) - 1, max(epochs) + 1)  
  
# 设置y轴的范围以适应所有数据  
plt.ylim(min(train_accuracies + test_accuracies + class_avg_mious + instance_avg_mious) - 0.05,  
         max(train_accuracies + test_accuracies + class_avg_mious + instance_avg_mious) + 0.05)  
  
# 显示网格  
plt.grid(True)  
  
# 显示图表  
plt.show()
########################################
plt.figure(figsize=(12, 6))  
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='^', linestyle='--')  
plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o', linestyle='-') 
plt.plot(epochs, best_accuracies, label='Best Accuracy', marker='^', linestyle='--', color='r')   

# 添加图例  
plt.legend()  

# 添加标题和坐标轴标签  
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')  
plt.ylabel('Accuracy')  
  
# 设置x轴的范围，确保所有点都可见  
plt.xlim(min(epochs) - 1, max(epochs) + 1)  
  
# 设置y轴的范围以适应所有数据  
plt.ylim(min(train_accuracies + test_accuracies + class_avg_mious + instance_avg_mious) - 0.05,  
         max(train_accuracies + test_accuracies + class_avg_mious + instance_avg_mious) + 0.05)  
# 显示网格  
plt.grid(True)  
  
# 显示图表  
plt.show()
########################################
plt.figure(figsize=(12, 6))  
plt.plot(epochs, learning_rates, label='Learning rate', marker='^', linestyle='--')  

# 在learning rate发生变化时标注出来  
for i in range(1, len(learning_rates)):  
    if learning_rates[i] != learning_rates[i-1]:  
        # 创建标注文本  
        annotation_text = f'Epoch {epochs[i]}: Learning Rate = {learning_rates[i]:.6f}'  
        plt.annotate(annotation_text,   
                     (epochs[i], learning_rates[i]),   
                     textcoords="offset points",   
                     xytext=(0,10),   
                     ha='center')  

# 添加图例  
plt.legend()  
  
# 添加标题和坐标轴标签  
plt.title('Epoch vs Learning rate')
plt.xlabel('Epoch')  
plt.ylabel('Learning rate')  
  
# 显示网格  
#plt.grid(True)  
  
# 显示图表  
plt.show()