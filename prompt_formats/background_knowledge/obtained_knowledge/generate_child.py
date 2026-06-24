import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'child'
queries_path = f"{dataset_name}"

variables = [
    AttrDict.make({
        "name": "BirthAsphyxia",
        "expression": "BirthAsphyxia",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Disease",
        "expression": "Disease",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "HypDistrib",
        "expression": "Hypoxemia Distribution",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LowerBodyO2",
        "expression": "Lower Body Oxygenation",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "HypoxiaInO2",
        "expression": "Hypoxia in Oxygen",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "RUQO2",
        "expression": "Right Upper Quadrant Oxygenation",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "CO2",
        "expression": "CO2",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CO2Report",
        "expression": "CO2 Report",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ChestXray",
        "expression": "Chest Xray",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "XrayReport",
        "expression": "X-ray Report",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Grunting",
        "expression": "Grunting",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "GruntingReport",
        "expression": "Grunting Report",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Age",
        "expression": "Age",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "LVH",
        "expression": "Left Ventricular Hypertrophy",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "DuctFlow",
        "expression": "Ductal Flow",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CardiacMixing",
        "expression": "Cardiac Blood Mixing",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LungParench",
        "expression": "Lung Parenchyma",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LungFlow",
        "expression": "LungFlow",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "Sick",
        "expression": "Sick",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LVHreport",
        "expression": "Left Ventricular Hypertrophy Report",
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
