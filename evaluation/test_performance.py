import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import json

def convert_results(test_results):
    """
    将测试结果转换为数字。
    - "yes" 转换为 1
    - "no" 转换为 0
    - 其他所有结果转换为 2
    """
    numeric_results = []
    for result in test_results:
        if result == 'Yes' or result == 'Yes.':
            numeric_results.append(1)
        elif result == 'No.' or result == 'No':
            numeric_results.append(0)
        else:
            numeric_results.append(0)
    return numeric_results
def calculate_SHD(graph1, graph2):
    """计算结构汉明距离（SHD）"""
    return np.sum(graph1 != graph2)

def calculate_SID(graph1, graph2):
    """计算结构干预距离（SID）的简化版本"""
    sid = 0
    n = graph1.shape[0]
    for i in range(n):
        parents1 = set(np.where(graph1[:, i])[0])
        parents2 = set(np.where(graph2[:, i])[0])
        if parents1 != parents2:
            sid += 1
    return sid

def calculate_edge_sparsity(graph):
    """计算边缘稀疏性"""
    n = graph.shape[0]  # 节点数量
    actual_edges = np.sum(graph != 0)  # 实际存在的边的数量
    max_possible_edges = n * (n - 1)  # 最大可能的边的数量
    sparsity = actual_edges / max_possible_edges
    return sparsity

# 假设我们有两个数组，一个是测试结果，一个是标签结果
def add_zero_column(matrix):
    """在每行的特定位置添加 0，并将矩阵扩展到 8x8"""
    new_matrix = np.zeros((8, 8))
    for i in range(8):
        new_matrix[i, :i] = matrix[i, :i]
        new_matrix[i, i+1:] = matrix[i, i:]
    return new_matrix

model = 'gpt_4'
dataset = 'asia'

labels = pd.read_csv(f'generate_label/label/{dataset}.csv')  # 标签结果

labels = labels.values

df = pd.read_csv(f'Response/asia_{model}.csv')

# 提取 'answer' 列
test_results = df['Answer']
print(test_results)
converted_results = convert_results(test_results.values)
print(converted_results)
arrays = np.array_split(converted_results, 5)
reshaped_arrays = [arr.reshape(-1, 7)[:8, :] for arr in arrays]

# 步骤 3: 在对角线上添加 # 创建 5 个随机的 8x7 矩阵作为示例

# 对每个矩阵应用 add_zero_column 函数
processed_matrices = [add_zero_column(matrix) for matrix in reshaped_arrays]

print(processed_matrices)

results = []

for i, matrix in enumerate(processed_matrices):
    # 示例计算，需要根据您的具体情况调整
    train = processed_matrices[i]


# 计算 F1 score 和 accuracy
    f1 = f1_score(labels.reshape(labels.size,1), train.reshape(train.size,1), average='macro')
    precision = precision_score(labels.reshape(labels.size,1), train.reshape(train.size,1), average='macro')
    recall = recall_score(labels.reshape(labels.size,1), train.reshape(train.size,1), average='macro')
    accuracy = accuracy_score(labels.reshape(labels.size,1), train.reshape(labels.size,1))

    print("F1 Score:", f1)
    print("Accuracy:", accuracy)

    learned_graph = train
    true_graph = labels

    shd = calculate_SHD(learned_graph, true_graph)
    sid = calculate_SID(learned_graph, true_graph)

    print("SHD:", shd)
    print("SID:", sid)




    # 假设 graph 是一个邻接矩阵
    graph = train

    sparsity = calculate_edge_sparsity(graph)

    print("Edge Sparsity:", sparsity)
    results.append({
        'F1 Score': float(f1),
        'Precision': float(precision),
        'Recall': float(recall),
        'Accuracy': float(accuracy),
        'SHD': int(shd),
        'SID': int(sid),
        'Edge Sparsity': float(sparsity)
    })


with open(f'results_{model}.json', 'w') as file:
    json.dump(results, file)

print("结果已保存为 JSON 文件。")


