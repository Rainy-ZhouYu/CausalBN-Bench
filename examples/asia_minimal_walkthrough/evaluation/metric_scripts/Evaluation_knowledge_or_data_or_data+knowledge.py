import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import os

def convert_results(test_results):
    numeric_results = []
    for result in test_results:
        if result == 'Yes' or result == 'Yes.':
            numeric_results.append(1)
        elif result == 'No.' or result == 'No':
            numeric_results.append(0)
        else:
            numeric_results.append(2)
    return numeric_results
def calculate_SHD(graph1, graph2):

    return np.sum(graph1 != graph2)

def calculate_SID(graph1, graph2):

    sid = 0
    n = graph1.shape[0]
    for i in range(n):
        parents1 = set(np.where(graph1[:, i])[0])
        parents2 = set(np.where(graph2[:, i])[0])
        if parents1 != parents2:
            sid += 1
    return sid

def calculate_edge_sparsity(graph):
    n = graph.shape[0]  # 节点数量
    actual_edges = np.sum(graph != 0)  # 实际存在的边的数量
    max_possible_edges = n * (n - 1)  # 最大可能的边的数量
    sparsity = actual_edges / max_possible_edges
    return sparsity

# 假设我们有两个数组，一个是测试结果，一个是标签结果
def add_zero_column(matrix):
    """在每行的特定位置添加 0，并将矩阵扩展到 8x8"""
    new_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(matrix.shape[0]):
        new_matrix[i, :i] = matrix[i, :i]
        new_matrix[i, i+1:] = matrix[i, i:]
    return new_matrix



def main(model, dataset, res, input_file_path):
    labels = pd.read_csv(f'generate_label/label/{dataset}.csv')  # 标签结果
    labels = labels.values
    with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    #df = pd.read_csv(f'Response/asia_{model}.csv')
    df = pd.read_csv(input_file_path)
    # 提取 'answer' 列
    df = pd.read_csv(input_file_path,encoding='utf-8')
    # 提取 'answer' 列
    test_results = df.iloc[:, 1]
    
    print(test_results)
    converted_results = convert_results(test_results.values)
    classes = int(converted_results.__len__() / (labels.shape[0] * (labels.shape[0]-1)))
    print(converted_results)
    arrays = np.array_split(converted_results, classes)
    reshaped_arrays = [arr.reshape(-1, labels.shape[0]-1)[:labels.shape[0], :] for arr in arrays]


    # 对每个矩阵应用 add_zero_column 函数
    processed_matrices = [add_zero_column(matrix) for matrix in reshaped_arrays]

    print(processed_matrices)

    results = []
    F1 = []
    ACC =[]
    SHD = []
    SID =[]
    Sparsity = []
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
        F1.append(f1)
        ACC.append(accuracy)
        SHD.append(shd)
        SID.append(sid)
        Sparsity.append(sparsity)
    f1_mean = np.mean(F1)
    f1_variance = np.var(F1)

    accuracy_mean = np.mean(ACC)
    accuracy_variance = np.var(ACC)
    SHD_mean = np.mean(SHD)
    SHD_variance = np.var(SHD)
    SID_mean = np.mean(SID)
    SID_variance = np.var(SID)
    Sparsity_mean = np.mean(Sparsity)
    Sparsity_variance = np.var(Sparsity)
    print("f1_mean:", f1_mean)
    print("F1 分数均值:", f1_variance)
    print("accuracy_mean:", accuracy_mean)
    print("accuracy 分数均值:", accuracy_variance)
    print("SHD_mean:", SHD_mean)
    print("SHD分数均值:", SHD_variance)
    print("SID mean:", SID_mean)
    print("SID 分数均值:", SID_variance)
    print("Sparsity mean:", Sparsity_mean)
    print("Sparsity 分数均值:", Sparsity_variance)
    res = res.append({
        "Model": f'{model}',
        "Dataset": f"{dataset}",
        "f1_mean": f1_mean,
        "f1_variance": f1_variance,
        "accuracy_mean": accuracy_mean,
        "accuracy_variance": accuracy_variance,
        "SHD_mean": SHD_mean,
        "SHD_variance": SHD_variance,
        "SID_mean": SID_mean,
        "SID_variance": SID_variance,
        "Sparsity_mean": Sparsity_mean,
        "Sparsity_variance": Sparsity_variance
    }, ignore_index=True)

    return res

if __name__ == '__main__':
    # models = ['LLAMA7B', 'LLAMA13B', 'OPT2d7b', 'OPT6d7b', 'Internlm7b', 'Internlm20b', 'falcon7b']#'falcon7b',
    # input_datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
    #             "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]#"sachs",
    # models = ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']#'falcon7b',
    # input_datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
    #             "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]#"sachs",
    models = ['gpt-4-1106-preview']#'falcon7b',
    input_datasets = ["insurance",]#"sachs",

    res = pd.DataFrame(columns=["Model", "Dataset", "f1_mean", "f1_variance", "accuracy_mean", "accuracy_variance",
                                "SHD_mean", "SHD_variance", "SID_mean", "SID_variance",
                                "Sparsity_mean", "Sparsity_variance"])
    for model in models:
        for input_dataset in input_datasets:
            input_file_path = f'Task_Data_Knowledge_Causality/{model}_{input_dataset}.csv'
            res = main(model, input_dataset, res, input_file_path)

    output_file_path = f'Result_GPT_Knowledge_{models}.csv'
    output_dir = os.path.dirname(output_file_path)

    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    res.to_csv(output_file_path, index=False)
