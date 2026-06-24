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
queries_path = f"./question/{dataset_name}"
# Age,GoodStudent,SocioEcon,RiskAversion,DrivingSkill,SeniorTrain,MedCost,VehicleYear,MakeModel,HomeBase,AntiTheft,OtherCar,DrivQuality,DrivHist,RuggedAuto,Antilock,CarValue,Airbag,ThisCarDam,ThisCarCost,OtherCarCost,Cushioning,Accident,ILiCost,Mileage,PropCost,Theft
# Age: 保险持有人或驾驶员的年龄。
# GoodStudent: 是否是表现良好的学生，这通常会影响保险费率。
# SocioEcon (Socioeconomic Status): 客户的社会经济状态，包括收入、教育水平等。
# RiskAversion: 对风险的厌恶程度，影响保险购买和选择。
# DrivingSkill: 驾驶技能水平。
# SeniorTrain: 针对老年人的驾驶培训参与情况。
# MedCost (Medical Cost): 预期的或历史的医疗费用。
# VehicleYear: 车辆的制造年份。
# MakeModel: 车辆的品牌和型号。
# HomeBase: 保险持有人的居住地，可能影响风险评估。
# AntiTheft: 车辆是否安装了防盗系统。
# OtherCar: 保险持有人是否拥有其他车辆。
# DrivQuality (Driving Quality): 驾驶质量评估。
# DrivHist (Driving History): 驾驶历史记录，包括违规和事故记录。
# RuggedAuto (Ruggedness of Automobile): 车辆的耐用性和可靠性。
# Antilock: 是否装有防抱死刹车系统（ABS）。
# CarValue: 车辆的市场价值。
# Airbag: 是否装有安全气囊。
# ThisCarDam (This Car Damage): 当前车辆的损坏情况。
# ThisCarCost (This Car Cost): 维护和修理当前车辆的费用。
# OtherCarCost: 维护和修理其他车辆的费用。
# Cushioning: 车辆的安全缓冲设施，如气囊等。
# Accident: 是否曾经发生过交通事故。
# ILiCost (Insurance Liability Cost): 保险责任成本，即保险公司可能承担的最大赔付额。
# Mileage: 车辆的行驶里程，通常影响车辆保险费用。
# PropCost (Property Cost): 与车辆相关的财产损失成本。
# Theft: 车辆被盗的风险或历史
def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    name_str = "Age,GoodStudent,SocioEcon,RiskAversion,DrivingSkill,SeniorTrain,MedCost,VehicleYear,MakeModel,HomeBase,AntiTheft,OtherCar,DrivQuality,DrivHist,RuggedAuto,Antilock,CarValue,Airbag,ThisCarDam,ThisCarCost,OtherCarCost,Cushioning,Accident,ILiCost,Mileage,PropCost,Theft"    # Splitting the string into a list and formatting each element
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