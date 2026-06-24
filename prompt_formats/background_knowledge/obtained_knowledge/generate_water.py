import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'water'
queries_path = f"{dataset_name}"

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    name_str = "C_NI_12_00,C_NI_12_15,CBODD_12_15,CKNI_12_00,CKNI_12_15,CKND_12_15,CBODD_12_00,CNOD_12_15,CBODN_12_15,CKND_12_00,CKNN_12_15,CNOD_12_00,CNON_12_15,CBODN_12_00,CKNN_12_00,CNON_12_00,C_NI_12_30,CBODD_12_30,CKNI_12_30,CKND_12_30,CNOD_12_30,CBODN_12_30,CKNN_12_30,CNON_12_30,C_NI_12_45,CBODD_12_45,CKNI_12_45,CKND_12_45,CNOD_12_45,CBODN_12_45,CKNN_12_45,CNON_12_45"
    # Splitting the string into a list and formatting each element"
    names = [f"{item}" for item in name_str.split(',')]

    # The final list
    expressions = names

    variables = [AttrDict.make({
        "name": name,
        "expression": expr,
        "singular": True,
        "optionalThe": True,
        "alt": []
    }) for name, expr in zip(names, expressions)]


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
