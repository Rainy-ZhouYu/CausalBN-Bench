import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'insurance'
queries_path = f"{dataset_name}"

variables = [
    AttrDict.make({
        "name": "Age",
        "expression": "Age",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "GoodStudent",
        "expression": "Good Student",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "SocioEcon",
        "expression": "Socio-economic Status",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "RiskAversion",
        "expression": "Risk Aversion",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "DrivingSkill",
        "expression": "Driving Skill",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "SeniorTrain",
        "expression": "Senior Train",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "MedCost",
        "expression": "Medical Cost",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "VehicleYear",
        "expression": "Vehicle Year",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "MakeModel",
        "expression": "Make Model",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "HomeBase",
        "expression": "Home Base",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "AntiTheft",
        "expression": "Anti-theft system",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "OtherCar",
        "expression": "Other Cars Involved In The Accident",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "DrivQuality",
        "expression": "Driving Quality",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "DrivHist",
        "expression": "Driving History",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "RuggedAuto",
        "expression": "Ruggedness of Automobile",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Antilock",
        "expression": "Antilock(ABS)",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CarValue",
        "expression": "value of the car",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Airbag",
        "expression": "Airbag",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "ThisCarDam",
        "expression": "damage to this car",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ThisCarCost",
        "expression": "costs for the insured car",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "OtherCarCost",
        "expression": "costs for the other car",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Cushioning",
        "expression": "Cushioning",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "Accident",
        "expression": "severity of the accident",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ILiCost",
        "expression": "Insurance Liability Cost",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Mileage",
        "expression": "Mileage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "PropCost",
        "expression": "Property Cost",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Theft",
        "expression": "Theft",
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
