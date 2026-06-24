import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'alarm'
queries_path = f"{dataset_name}"

variables = [
    AttrDict.make({
        "name": "HYPOVOLEMIA",
        "expression": "HYPOVOLEMIA",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LVEDVOLUME",
        "expression": "Left Ventricular End-Diastolic Volume",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "STROKEVOLUME",
        "expression": "STROKEVOLUME",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CVP",
        "expression": "Central Venous Pressure",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PCWP",
        "expression": "Pulmonary Capillary Wedge Pressure",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LVFAILURE",
        "expression": "Left Ventricular Failure",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "HISTORY",
        "expression": "HISTORY",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CO",
        "expression": "Cardiac Output",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ERRLOWOUTPUT",
        "expression": "ERRLOWOUTPUT",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "HRBP",
        "expression": "HRBP",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ERRCAUTER",
        "expression": "ERRCAUTER",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "HREKG",
        "expression": "Heart Rate EKG",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "HRSAT",
        "expression": "Heart Rate Saturation",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "INSUFFANESTH",
        "expression": "Insufficient Anesthesia",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CATECHOL",
        "expression": "CATECHOL",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ANAPHYLAXIS",
        "expression": "ANAPHYLAXIS",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "TPR",
        "expression": "Total Peripheral Resistance",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "BP",
        "expression": "Blood Pressure",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "KINKEDTUBE",
        "expression": "KINKEDTUBE",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PRESS",
        "expression": "PRESS",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "VENTLUNG",
        "expression": "VENTLUNG",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "FIO2",
        "expression": "Fraction of Inspired Oxygen",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PVSAT",
        "expression": "Peripheral Venous Saturation",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),




    AttrDict.make({
        "name": "SAO2",
        "expression": "Arterial Oxygen Saturation",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PULMEMBOLUS",
        "expression": "Pulmonary Embolus",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PAP",
        "expression": "Pulmonary Artery Pressure",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "SHUNT",
        "expression": "SHUNT",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "INTUBATION",
        "expression": "INTUBATION",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "MINVOL",
        "expression": "Minimum Volume",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "VENTALV",
        "expression": "Ventilation Alveolar",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "DISCONNECT",
        "expression": "DISCONNECT",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

# VENTALV
# ARTCO2
# EXPCO2
# ERRCAUTER
# HREKG
# BP
    AttrDict.make({
        "name": "VENTTUBE",
        "expression": "Ventilation Tube",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "MINVOLSET",
        "expression": "Minimum Volume Set",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "VENTMACH",
        "expression": "Ventilation Machine",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "EXPCO2",
        "expression": "Expired CO2",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "ARTCO2",
        "expression": "Arterial CO2",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "HR",
        "expression": "Heart Rate",
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

    # 将DataFrame保存到新的csv中
    df_seventh.to_csv(f'Knowledge_{dataset_name}.csv', index=False)

if __name__ == "__main__":
    main()
