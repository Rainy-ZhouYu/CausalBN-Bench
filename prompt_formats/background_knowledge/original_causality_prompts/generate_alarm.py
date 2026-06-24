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
queries_path = f"./question/{dataset_name}"
# HYPOVOLEMIA,LVEDVOLUME,STROKEVOLUME,CVP,PCWP,LVFAILURE,HISTORY,CO,ERRLOWOUTPUT,HRBP,ERRCAUTER,HREKG,HRSAT,INSUFFANESTH,CATECHOL,ANAPHYLAXIS,TPR,BP,KINKEDTUBE,PRESS,VENTLUNG,FIO2,PVSAT,SAO2,PULMEMBOLUS,PAP,SHUNT,INTUBATION,MINVOL,VENTALV,DISCONNECT,VENTTUBE,MINVOLSET,VENTMACH,EXPCO2,ARTCO2,HR
# HR (Heart Rate)：心率。
# BP (Blood Pressure)：血压。
# CO (Cardiac Output)：心脏输出量。
# PAP (Pulmonary Artery Pressure)：肺动脉压。
# CVP (Central Venous Pressure)：中心静脉压。
# PCWP (Pulmonary Capillary Wedge Pressure)：肺毛细血管楔压。
# TPR (Total Peripheral Resistance)：全身外周阻力。
# MinVol (Minimum Volume)：最小容积。
# FIO2 (Fraction of Inspired Oxygen)：吸入氧气浓度。
# KINKEDTUBE (Kinked Tube)：管道是否扭曲。
# VENTMACH (Ventilation Machine)：通风机状态。
# VENTLUNG (Ventilation Lung)：肺部通风情况。
# DISCONNECT (Disconnect)：是否断开连接。
# ANAPHYLAXIS (Anaphylaxis)：过敏反应。
# INTUBATION (Intubation)：插管情况。
# PVSAT (Peripheral Venous Saturation)：外周静脉饱和度。
# VENTTUBE (Ventilation Tube)：通风管状态。

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
    txt_file_path = f'question/questions_{dataset_name}.txt'
    csv_file_path = f'question/questions_{dataset_name}.csv'
    txt_to_csv(txt_file_path, csv_file_path)
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ['prompt'] + df.columns.tolist()[1:]

    # 将DataFrame保存到新的csv中
    df.to_csv(f'question/questions_{dataset_name}.csv', index=False)

if __name__ == "__main__":
    main()
