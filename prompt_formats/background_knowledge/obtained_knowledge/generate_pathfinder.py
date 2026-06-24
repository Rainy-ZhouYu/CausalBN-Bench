import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'pathfinder'
queries_path = f"{dataset_name}"

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    name_str = "Fault,F1,F97,F2,F78,F3,F4,F5,F53,F6,F7,F56,F8,F9,F10,F55,F52,F11,F12,F13,F14,F15,F16,F17,F18,F19,F41,F44,F20,F90,F21,F22,F23,F24,F25,F26,F27,F28,F92,F98,F30,F31,F32,F33,F34,F35,F36,F37,F84,F96,F38,F39,F40,F42,F43,F45,F46,F47,F85,F48,F49,F50,F51,F83,F54,F57,F58,F59,F60,F61,F62,F63,F64,F65,F66,F67,F68,F69,F72,F70,F71,F73,F74,F75,F76,F77,F79,F80,F81,F82,F87,F88,F89,F91,F93,F94,F95,F99,F100,F105,F101,F102,F103,F104,F106,F107,F108,F86,F29"
    # Splitting the string into a list and formatting each element
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
