from pgmpy.models import BayesianNetwork
from pgmpy.independencies import Independencies
from pgmpy.readwrite import BIFReader
# 创建贝叶斯网络
file_path = 'Bif_File/asia.bif'

# 读取.bif文件
reader = BIFReader(file_path)

# 获取模型
model = reader.get_model()

bn = BayesianNetwork(model)

# 计算条件独立性
independencies = bn.get_independencies()
independencies_list = independencies.get_assertions()
print(independencies_list)
# 为了简化问题，我们将集中在直接的相关性（一步因果关系）而非间接相关性
# 对于每个节点 A，找出与其直接相关的节点
structure = model.edges()
print("因果骨架（边）:")
print(structure)
direct_relations = {}
for node in structure:
    direct_relations[node] = set()
    for assertion in independencies_list:
        if assertion[0] == node and not assertion[2]:
            # 如果节点 A 与节点 B 直接相关（不考虑任何条件）
            direct_relations[node].add(assertion[1])

# 检查每个节点 A，找到满足条件的节点 B
output_nodes = []
for node_a in structure:
    for node_b in structure:
        if node_b != node_a and node_b not in direct_relations[node_a]:
            # 如果节点 B 与节点 A 不直接相关
            for node_c in structure:
                if node_c != node_a and node_c in direct_relations[node_b] and node_c in direct_relations[node_a]:
                    # 如果节点 C 与节点 B 直接相关，且节点 C 也与节点 A 直接相关
                    output_nodes.append((node_a, node_b, node_c))
                    break  # 找到满足条件的节点后跳出循环

output_nodes  # 输出满足条件的节点组合列表 (节点 A, 节点 B, 节点 C)
