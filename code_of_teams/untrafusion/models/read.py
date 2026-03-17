import torch

# 加载权重文件
path = './model_zoo/06_spynet.pth'
state_dict = torch.load(path, map_location='cpu')

# 如果权重文件被包裹在 'params' 或 'net' 键下（某些框架的习惯）
if 'params' in state_dict:
    state_dict = state_dict['params']

# 打印所有的键名
for key in state_dict.keys():
    print(key)