from pgmpy.readwrite import BIFReader
from pgmpy.models import BayesianModel
import pandas as pd


file_path = 'alarm.bif'
# 读取 .bif 文件
# 替换为您的 BIF 文件路径

# 初始化一个字典来存储信息
bif_data = {
    "network": "",
    "variables": {},
    "probabilities": {}
}

# 读取 BIF 文件
with open(file_path, 'r') as file:
    lines = file.readlines()

processed_lines = []
for line in lines:
    if line.strip() == "}":
        processed_lines[-1] = processed_lines[-1].strip() + " }"
    else:
        processed_lines.append(line.strip())

processed_lines1 = []
i = 0
while i < len(processed_lines):
    line = processed_lines[i].strip()
    if line.endswith("{") and i + 1 < len(processed_lines):
        # 合并当前行和下一行
        next_line = processed_lines[i + 1].strip()
        processed_lines1.append(line + " " + next_line)
        i += 2  # 跳过下一行
    else:
        processed_lines1.append(line)
        i += 1

# 解析 BIF 文件内容
for line in processed_lines1:
    # line = processed_lines1.strip()
    if line.startswith("network"):
        bif_data["network"] = line
    elif line.startswith("variable"):
        # 这里可以添加解析变量定义的逻辑
        pass
    elif line.startswith("probability"):
        # 这里可以添加解析概率分布的逻辑
        pass


model = processed_lines1.get_model()
# 打印结果以检查
print(bif_data)
