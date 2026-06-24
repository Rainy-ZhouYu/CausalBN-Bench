import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
queries_path = "./question/asia"

# asia
# p;''''tub
# either
# xray
# lung
# smoke
# bronc
# dysp
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
        "expression": "Asia",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Tub",
        "expression": "Tub",
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
        "name": "Xray",
        "expression": "Xray",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Lung",
        "expression": "Lung",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Smoke",
        "expression": "Smoke",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Bronc",
        "expression": "Bronc",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Dysp",
        "expression": "Dysp",
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
    txt_file_path = 'question/questions_asia.txt'
    csv_file_path = 'question/questions_asia.csv'
    txt_to_csv(txt_file_path, csv_file_path)
    df = pd.read_csv('question/questions_asia.csv', header=None)
    df.columns = ['prompt'] + df.columns.tolist()[1:]

    # 将DataFrame保存回CSV文件
    df.to_csv('question/questions_asia_v2.csv', index=False)
if __name__ == "__main__":
    main()
