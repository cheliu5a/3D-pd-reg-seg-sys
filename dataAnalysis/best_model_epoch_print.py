import re  
  
# 定义日志文件路径  
#log_file_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/log/classification/pointnet2_msg_normals/logs/pointnet2_cls_msg.txt'  
log_file_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/log/part_seg/pointnet2_part_seg_msg/logs/pointnet2_part_seg_msg.txt'
# 读取日志文件内容  
with open(log_file_path, 'r') as file:  
    log_content = file.read()  
  
# 正则表达式用于匹配 "Save model..." 和 "Epoch" 行  
save_model_pattern = re.compile(r".*- Model - INFO - Save model.*\n")  
epoch_pattern = re.compile(r"Epoch (\d+) \S*:")  
  
# 查找所有 "Save model..." 行  
save_model_lines = save_model_pattern.findall(log_content)  

# 用于存储所有找到的 epochs  
all_epochs = set()  
  
# 遍历每个 "Save model..." 行  
for save_line in save_model_lines:  
    # 从当前 "Save model..." 行开始，向前搜索最近的 "Epoch" 行  
    start_index = log_content.rindex(save_line)  
      
    # 将 finditer 对象转换为列表，然后反转这个列表  
    epoch_matches = list(epoch_pattern.finditer(log_content[:start_index]))[::-1]  
      
    # 遍历反转后的列表，找到最近的 "Epoch" 行  
    for match in epoch_matches:  
        # 找到一个 "Epoch" 行后，将其 epoch 编号添加到集合中  
        all_epochs.add(int(match.group(1)))  
        # 停止搜索，因为我们已经找到了最近的 "Epoch" 行  
        break  
  
# 打印所有满足条件的 epochs  
print("Epochs with 'Save model...' messages:")  
for epoch in sorted(all_epochs):  
    print(epoch)