from pgmpy.readwrite import BIFReader


# 替换为您的.bif文件路径
file_path = 'Bif_File/asia.bif'

# 读取.bif文件
reader = BIFReader(file_path)

# 获取模型
model = reader.get_model()

# 提取因果骨架（网络结构）
structure = model.edges()
print("因果骨架（边）:")
print(structure)


print("\n节点的条件概率表:")
for node in model.nodes():
    cpd = model.get_cpds(node)
    print("节点:", node)
    print("条件概率表:")
    print(cpd)


