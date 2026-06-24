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
queries_path = f"./question/{dataset_name}"
# BirthAsphyxia,Disease,HypDistrib,LowerBodyO2,HypoxiaInO2,RUQO2,CO2,CO2Report,ChestXray,XrayReport,Grunting,GruntingReport,Age,LVH,DuctFlow,CardiacMixing,LungParench,LungFlow,Sick,LVHreport
# BirthAsphyxia (出生窒息): 新生儿出生时是否经历了窒息。
# Disease (疾病): 新生儿可能患有的疾病。
# HypDistrib (Hypoxemia Distribution): 低氧血症在身体不同部位的分布。
# LowerBodyO2 (Lower Body Oxygenation): 下半身的氧合情况。
# HypoxiaInO2 (Hypoxia in Oxygen): 在吸氧条件下的低氧状况。
# RUQO2 (Right Upper Quadrant Oxygenation): 右上腹部的氧合情况。
# CO2: 血液中的二氧化碳水平。
# CO2Report (CO2 Report): 有关二氧化碳水平的报告。
# ChestXray: 胸部X光检查。
# XrayReport (X-ray Report): X光检查报告。
# Grunting: 新生儿的呻吟呼吸声，可能是呼吸困难的迹象。
# GruntingReport (Grunting Report): 有关呻吟呼吸的报告。
# Age: 新生儿的年龄或出生时长。
# LVH (Left Ventricular Hypertrophy): 左心室肥大。
# DuctFlow (Ductal Flow): 导管流动情况，可能指动脉导管的血流。
# CardiacMixing (Cardiac Blood Mixing): 心内血液混合情况。
# LungParench (Lung Parenchyma): 肺实质，肺组织的功能部分。
# LungFlow: 肺部血流情况。
# Sick: 新生儿是否生病或不适。
# LVHreport (Left Ventricular Hypertrophy Report): 关于左心室肥大的报告。

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    name_str = "BirthAsphyxia,Disease,HypDistrib,LowerBodyO2,HypoxiaInO2,RUQO2,CO2,CO2Report,ChestXray,XrayReport,Grunting,GruntingReport,Age,LVH,DuctFlow,CardiacMixing,LungParench,LungFlow,Sick,LVHreport"    # Splitting the string into a list and formatting each element
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
    txt_file_path = f'question/questions_{dataset_name}.txt'
    csv_file_path = f'question/questions_{dataset_name}.csv'
    txt_to_csv(txt_file_path, csv_file_path)
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ['prompt'] + df.columns.tolist()[1:]

    # 将DataFrame保存到新的csv中
    df.to_csv(f'question/questions_{dataset_name}.csv', index=False)

if __name__ == "__main__":
    main()
