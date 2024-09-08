import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import os

def read_first_column(csv_file):
    return pd.read_csv(csv_file, usecols=[0])

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
def adjust_dimensions(col1, col2):
    len1 = len(col1)
    len2 = len(col2)
    if len1 > len2:
        return col1[:len2], col2
    elif len2 > len1:
        return col1, col2[:len1]
    else:
        return col1, col2
def main(model, dataset, res, input_file_path):
    labels = pd.read_csv(f'{input_dataset}.csv')
    with open(input_file_path, 'r', encoding='gbk', errors='ignore') as f:
        content = f.read()

    with open(input_file_path, 'w', encoding='gbk') as f:
        f.write(content)
    labels = labels.iloc[:, 1]
    labels = labels.values
    df = pd.read_csv(input_file_path, encoding='gbk')

    test_results = df.iloc[:, 1]
    print(test_results)
    converted_results = convert_results(test_results.values)
    converted_results = np.array(converted_results)
    adjusted_col1, adjusted_col2 = adjust_dimensions(labels, converted_results)


    f1 = f1_score(adjusted_col1, adjusted_col2, average='macro')
    accuracy = accuracy_score(adjusted_col1, adjusted_col2)
    res = res.append({
        "Model": f'{model}',
        "Dataset": f"{dataset}",
        "f1": f1,
        "accuracy": accuracy,
    }, ignore_index=True)

    return res
if __name__ == '__main__':
    models = ['gpt-4-0613']
    input_datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley",]#
    res = pd.DataFrame(columns=["Model", "Dataset", "f1", "accuracy"])
    for model in models:
        for input_dataset in input_datasets:
            input_file_path = f'Task_Skeleton_Process/{model}_{input_dataset}.csv'


            res = main(model, input_dataset, res, input_file_path)

    output_file_path = f'Result_GPT_skeleton_{models}.csv'
    output_dir = os.path.dirname(output_file_path)

    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    res.to_csv(output_file_path, index=False)