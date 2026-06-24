import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'asia'
queries_path = f"./Ranking/{dataset_name}"

variables = [
    AttrDict.make({
        "name": "Asia",
        "expression": "Visiting to Asia",
        "singular": True,
        "optionalThe": False,
        "alt": []
    }),
    AttrDict.make({
        "name": "Tub",
        "expression": "Tuberculosis",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Either",
        "expression": "Either",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Smoke",
        "expression": "Smoking",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Lung",
        "expression": "Lung cancer",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Bronc",
        "expression": "Bronchitis",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Dysp",
        "expression": "Dyspnoea",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Xray",
        "expression": "X-ray",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
]

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    question_instances = instantiate_questions(questions, variables)
    if not dry_run:
        store_query_instances(queries_path, question_instances)
    print("done.")
    txt_file_path = f'Ranking/ranking_{dataset_name}.txt'
    csv_file_path = f'Ranking/ranking_{dataset_name}.csv'
    txt_to_csv(txt_file_path, csv_file_path)
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ['prompt'] + df.columns.tolist()[1:]
    # combined_column = df.iloc[:, 0].astype(str) + ", " + df.iloc[:, 1].astype(str)
    #
    # # 创建一个新的 DataFrame，包含合并后的列和原始 DataFrame 的其他列
    # new_df = pd.DataFrame(combined_column, columns=['prompt'])
    # df['prompt'] = df.iloc[:, 0].astype(str) + "," + df.iloc[:, 1].astype(str)
    num = variables.__len__()
    df_seventh = df.iloc[::num-1]

    # 将DataFrame保存到新的csv中
    df_seventh.to_csv(f'Ranking/ranking_{dataset_name}.csv', index=False)

if __name__ == "__main__":
    main()
