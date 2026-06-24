import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False

dataset_name = 'earthquake'
queries_path = f"./Ranking/{dataset_name}"
# Burglary,Alarm,Earthquake,JohnCalls,MaryCalls
variables = [
    AttrDict.make({
        "name": "Burglary",
        "expression": "Burglary",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Alarm",
        "expression": "Alarm",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Earthquake",
        "expression": "Earthquake",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "JohnCalls",
        "expression": "JohnCalls",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "MaryCalls",
        "expression": "MaryCalls",
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
    num = variables.__len__()
    df_seventh = df.iloc[::num-1]
    df_seventh.to_csv(f'Ranking/ranking_{dataset_name}.csv', index=False)
if __name__ == "__main__":
    main()
