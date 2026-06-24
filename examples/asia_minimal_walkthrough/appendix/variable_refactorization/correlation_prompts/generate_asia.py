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
queries_path = f"./question/{dataset_name}"

# asia	tub	either	smoke	lung	bronc	dysp	xray
# 吸烟（Smoking）：表示个体是否吸烟的变量。
# 肺癌（Lung Cancer）：表示个体是否患有肺癌的变量。
# 支气管炎（Bronchitis）：表示个体是否患有支气管炎的变量。
# 肺结核（Tuberculosis）：表示个体是否患有肺结核的变量。
# 肺结核或肺癌（Either）：一个复合变量，表示个体是否患有肺结核或肺癌。
# X射线检查（X-ray）：表示X射线检查结果的变量。
# 呼吸困难（Dyspnoea）：表示个体是否有呼吸困难的症状的变量。
# 访问亚洲（Visit to Asia）：一个与旅行史相关的变量，表示个体是否访问过亚洲。
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
    txt_file_path = f'question/questions_{dataset_name}.txt'
    csv_file_path = f'question/questions_{dataset_name}.csv'
    txt_to_csv(txt_file_path, csv_file_path)
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ['prompt'] + df.columns.tolist()[1:]

    # 将DataFrame保存到新的csv中
    df.to_csv(f'question/questions_{dataset_name}.csv', index=False)
if __name__ == "__main__":
    main()
