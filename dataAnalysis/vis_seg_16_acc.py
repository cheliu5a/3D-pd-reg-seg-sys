import re  
from collections import defaultdict  
  
# 日志文件路径  
log_file_path = '/home/chen/Pointnet_Pointnet2_pytorch-master/log/part_seg/pointnet2_part_seg_msg/logs/pointnet2_part_seg_msg.txt'  
  
# 用于存储每个类别的最大mIoU值的字典  
max_miou_per_category = defaultdict(float)  
  
# 正则表达式模式，用于匹配eval mIoU of后面的类别和值  
pattern = r"eval mIoU of (.*?)\s+(\d+\.\d+)"  
  
# 打开日志文件并读取内容  
with open(log_file_path, 'r') as file:  
    log_content = file.read()  
  
    # 查找所有匹配项  
    matches = re.findall(pattern, log_content)  
  
    # 遍历匹配项，更新每个类别的最大mIoU值  
    for match in matches:  
        category, miou = match  
        # 将mIoU值转换为浮点数  
        miou_value = float(miou)  
        # 更新这个类别的最大mIoU值  
        max_miou_per_category[category] = max(miou_value, max_miou_per_category[category])  
  
# 打印每个类别的最大mIoU值  
for category, max_miou in max_miou_per_category.items():  
    print(f"Max mIoU for {category}: {max_miou}")  
  
# 如果你想要找到所有epochs中所有数据的最大值，可以这样做：  
all_miou_values = list(max_miou_per_category.values())  
max_miou_overall = max(all_miou_values)  
print(f"Overall max mIoU: {max_miou_overall}")
'''
Max mIoU for Airplane: 0.833076
Max mIoU for Bag: 0.846507
Max mIoU for Cap: 0.880533
Max mIoU for Car: 0.785224
Max mIoU for Chair: 0.908337
Max mIoU for Earphone: 0.798434
Max mIoU for Guitar: 0.914526
Max mIoU for Knife: 0.881099
Max mIoU for Lamp: 0.855879
Max mIoU for Laptop: 0.95683
Max mIoU for Motorbike: 0.734419
Max mIoU for Mug: 0.957847
Max mIoU for Pistol: 0.83423
Max mIoU for Rocket: 0.635665
Max mIoU for Skateboard: 0.768516
Max mIoU for Table: 0.832732
Overall max mIoU: 0.957847
'''