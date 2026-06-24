import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'sachs'
queries_path = f"{dataset_name}"

variables = [
    AttrDict.make({
        "name": "Erk",
        "expression": "Extracellular signal-regulated kinases",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Akt",
        "expression": "Protein Kinase B",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Mek",
        "expression": "Mitogen-Activated Protein Kinase Kinase",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PIP3",
        "expression": "Phosphatidylinositol 3 4 and 5 trisphosphate",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PIP2",
        "expression": "Phosphatidylinositol 4 and 5 bisphosphate",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PKA",
        "expression": "Protein Kinase A",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Jnk",
        "expression": "c-Jun N-terminal Kinases",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "P38",
        "expression": "P38 Mitogen-Activated Protein Kinases",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Raf",
        "expression": "Rapidly Accelerated Fibrosarcoma",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PKC",
        "expression": "Protein Kinase C",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Plcg",
        "expression": "Phospholipase C gamma",
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
    txt_file_path = f'Knowledge_{dataset_name}.txt'
    csv_file_path = f'Knowledge_{dataset_name}.csv'
    txt_to_csv(txt_file_path, csv_file_path)
    df = pd.read_csv(csv_file_path, header=None)
    # df.columns = ['prompt'] + df.columns.tolist()[1:]
    combined_column = df.iloc[:, 0].astype(str) + ", " + df.iloc[:, 1].astype(str)

    new_df = pd.DataFrame(combined_column, columns=['prompt'])
    num = variables.__len__()
    df_seventh = new_df.iloc[::num-1]

    df_seventh.to_csv(f'Knowledge_{dataset_name}.csv', index=False)

if __name__ == "__main__":
    main()
