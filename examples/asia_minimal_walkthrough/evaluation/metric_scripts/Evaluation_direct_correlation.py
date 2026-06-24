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
    n = graph.shape[0]
    actual_edges = np.sum(graph != 0)
    max_possible_edges = n * (n - 1)
    sparsity = actual_edges / max_possible_edges
    return sparsity

def add_zero_column(matrix):
    new_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(matrix.shape[0]):
        new_matrix[i, :i] = matrix[i, :i]
        new_matrix[i, i+1:] = matrix[i, i:]
    return new_matrix



def main(model, dataset, res, input_file_path):
    labels = pd.read_csv(f'label/{dataset}.csv')  # 标签结果
    with open(input_file_path, 'r', encoding='gbk', errors='ignore') as f:
        content = f.read()

    with open(input_file_path, 'w', encoding='gbk') as f:
        f.write(content)
    labels = labels.values
    transpose = labels.T

    symmetric_matrix = labels + transpose


    symmetric_matrix[symmetric_matrix > 1] = 1

    print(symmetric_matrix)
    labels = symmetric_matrix
    df = pd.read_csv(input_file_path,encoding='gbk')

    test_results = df.iloc[:, 1]
    print(test_results)
    converted_results = convert_results(test_results.values)
    classes = int(converted_results.__len__() / (labels.shape[0] * (labels.shape[0]-1)))
    print(converted_results)
    arrays = np.array_split(converted_results, classes)
    reshaped_arrays = [arr.reshape(-1, labels.shape[0]-1)[:labels.shape[0], :] for arr in arrays]


    processed_matrices = [add_zero_column(matrix) for matrix in reshaped_arrays]

    print(processed_matrices)

    results = []
    F1 = []
    ACC =[]
    SHD = []
    SID =[]
    Sparsity = []
    for i, matrix in enumerate(processed_matrices):

        train = processed_matrices[i]



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
    print("f1_mean:", F1)
    print("F1 分数均值:", F1)
    print("accuracy_mean:", ACC)
    print("accuracy 分数均值:", ACC)
    print("SHD_mean:", SHD)
    print("SHD分数均值:", SHD)
    print("SID mean:", SID)
    print("SID 分数均值:", SID)
    print("Sparsity mean:", Sparsity)
    print("Sparsity 分数均值:", Sparsity)
    res = res.append({
        "Model": f'{model}',
        "Dataset": f"{dataset}",
        "f1_mean": F1,
        "f1_variance": F1,
        "accuracy_mean": ACC,
        "accuracy_variance": ACC,
        "SHD_mean": SHD,
        "SHD_variance": SHD,
        "SID_mean": SID,
        "SID_variance": SID,
        "Sparsity_mean": Sparsity,
        "Sparsity_variance": Sparsity
    }, ignore_index=True)


    return res

if __name__ == '__main__':
    models = ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106']
    input_datasets = ["asia", "cancer", "earthquake", "sachs", "water", "alarm", "survey", "child",
                      "insurance", "mildew", "hepar2"]

    res = pd.DataFrame(columns=["Model", "Dataset", "f1_mean", "f1_variance", "accuracy_mean", "accuracy_variance",
                                "SHD_mean", "SHD_variance", "SID_mean", "SID_variance",
                                "Sparsity_mean", "Sparsity_variance"])
    for model in models:
        for input_dataset in input_datasets:
            input_file_path = f'Task_Relation/{model}_{input_dataset}.csv'


            res = main(model, input_dataset, res, input_file_path)

    output_file_path = f'Relation_{models}.csv'
    output_dir = os.path.dirname(output_file_path)

    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    res.to_csv(output_file_path, index=False)