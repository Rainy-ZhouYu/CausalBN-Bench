import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'hailfinder'
queries_path = f"./question/{dataset_name}"
# alcoholism,THepatitis,Steatosis,vh_amn,ChHepatitis,hbsag,hbsag_anti,hbc_anti,hcv_anti,hbeag,hepatotoxic,RHepatitis,fatigue,phosphatase,inr,hepatomegaly,alt,ast,ggtp,anorexia,nausea,spleen,hospital,injections,transfusion,surgery,gallstones,choledocholithotomy,bilirubin,upper_pain,fat,pressure_ruq,flatulence,amylase,fibrosis,ESR,cholesterol,sex,PBC,Hyperbilirubinemia,age,ama,le_cells,joints,pain,platelet,encephalopathy,carcinoma,Cirrhosis,diabetes,obesity,triglycerides,pain_ruq,proteins,edema,alcohol,spiders,albumin,edge,irregular_liver,palms,itching,skin,jaundice,ascites,bleeding,urea,density,consciousness,hepatalgia
# N0_7muVerMo: 可能是指在 0 到 7 µm 高度范围内的垂直水汽含量。
# CombVerMo: 综合垂直水汽含量。
# SubjVertMo: 主观评估的垂直水汽含量。
# QGVertMotion: 准地转垂直运动。
# AreaMeso_ALS: 在某一区域内的中尺度气旋或类似结构。
# AreaMoDryAir: 干空气区域范围。
# CldShadeOth: 云遮蔽的其他指标。
# CompPlFcst: 综合平原地区预报。
# SatContMoist: 卫星连续水汽监测。
# CombMoisture: 综合湿度。
# RaoContMoist: 可能指雷欧连续水汽监测。
# VISCloudCov: 可见光云覆盖。
# CombClouds: 综合云量。
# IRCloudCover: 红外云覆盖。
# InsInMt: 山区的瞬时不稳定性。
# AMInstabMt: 山区上午的大气不稳定性。
# OutflowFrMt: 山区外流。
# CldShadeConv: 对流云遮蔽。
# MountainFcst: 山区天气预报。
# WndHodograph: 风向螺旋图。
# Boundaries: 天气边界。
# MorningBound: 早晨天气边界。
# CapChange: 锋盖变化。
# InsChange: 不稳定性变化。
# CapInScen: 锋盖在特定情景中。
# LoLevMoistAd: 低层湿度调整。
# InsSclInScen: 情景中的规模不稳定性。
# R5Fcst: R5 预报。
# Date: 日期。
# Scenario: 情景。
# ScenRelAMCIN: 情景中上午 CIN 的相关性。
# ScenRelAMIns: 情景中上午不稳定性的相关性。
# ScenRel3_4: 情景中 3 到 4 的相关性。
# ScnRelPlFcst: 情景中平原预报的相关性。
# Dewpoints: 露点。
# LowLLapse: 低层递减率。
# MeanRH: 平均相对湿度。
# MidLLapse: 中层递减率。
# MvmtFeatures: 运动特征。
# RHRatio: 相对湿度比率。
# SfcWndShfDis: 地表风切变距离。
# SynForcng: 大尺度强迫。
# TempDis: 温度分散。
# WindAloft: 高空风。
# WindFieldMt: 山区风场。
# WindFieldPln: 平原地区风场。
# AMCINInScen: 上午 CIN 在情景中。
# MorningCIN: 早晨 CIN。
# PlainsFcst: 平原地区预报。
# AMInsWliScen: 情景中上午不稳定性的概率。


def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    name_str = "N0_7muVerMo,CombVerMo,SubjVertMo,QGVertMotion,AreaMeso_ALS,AreaMoDryAir,CldShadeOth,CompPlFcst,SatContMoist,CombMoisture,RaoContMoist,VISCloudCov,CombClouds,IRCloudCover,InsInMt,AMInstabMt,OutflowFrMt,CldShadeConv,MountainFcst,WndHodograph,Boundaries,MorningBound,CapChange,InsChange,CapInScen,LoLevMoistAd,InsSclInScen,R5Fcst,Date,Scenario,ScenRelAMCIN,ScenRelAMIns,ScenRel3_4,ScnRelPlFcst,Dewpoints,LowLLapse,MeanRH,MidLLapse,MvmtFeatures,RHRatio,SfcWndShfDis,SynForcng,TempDis,WindAloft,WindFieldMt,WindFieldPln,AMCINInScen,MorningCIN,PlainsFcst,AMInsWliScen,LIfr12ZDENSd,AMDewptCalPl,N34StarFcst,LatestCIN,CurPropConv,LLIW"    # Splitting the string into a list and formatting each element
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

