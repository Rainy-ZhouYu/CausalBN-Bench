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
queries_path = f"./Ranking/{dataset_name}"
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

variables = [
    AttrDict.make({
        "name": "N0_7muVerMo",
        "expression": "10.7mu vertical motion",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CombVerMo",
        "expression": "Combined vertical motion",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "SubjVertMo",
        "expression": "Subjective judgment of vertical motion",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
# CldShadeOth,CompPlFcst,SatContMoist,CombMoisture,RaoContMoist,VISCloudCov,CombClouds,IRCloudCover,InsInMt,AMInstabMt,OutflowFrMt,CldShadeConv,MountainFcst,WndHodograph,Boundaries,MorningBound,CapChange,InsChange,CapInScen,LoLevMoistAd,InsSclInScen,R5Fcst,Date,Scenario,ScenRelAMCIN,ScenRelAMIns,ScenRel3_4,ScnRelPlFcst,Dewpoints,LowLLapse,MeanRH,MidLLapse,MvmtFeatures,RHRatio,SfcWndShfDis,SynForcng,TempDis,WindAloft,WindFieldMt,WindFieldPln,AMCINInScen,MorningCIN,PlainsFcst,AMInsWliScen,LIfr12ZDENSd,AMDewptCalPl,N34StarFcst,LatestCIN,CurPropConv,LLIW

    AttrDict.make({
        "name": "QGVertMotion",
        "expression": "Quasigeostrophic vertical motion",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "AreaMeso_ALS",
        "expression": "Area of meso-alpha",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "AreaMoDryAir",
        "expression": "Area of moisture and adry air",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
# CldShadeOth,CompPlFcst,SatContMoist,CombMoisture,RaoContMoist,VISCloudCov,CombClouds,IRCloudCover,InsInMt,AMInstabMt,OutflowFrMt,CldShadeConv,MountainFcst,WndHodograph,Boundaries,MorningBound,CapChange,InsChange,CapInScen,LoLevMoistAd,InsSclInScen,R5Fcst,Date,Scenario,ScenRelAMCIN,ScenRelAMIns,ScenRel3_4,ScnRelPlFcst,Dewpoints,LowLLapse,MeanRH,MidLLapse,MvmtFeatures,RHRatio,SfcWndShfDis,SynForcng,TempDis,WindAloft,WindFieldMt,WindFieldPln,AMCINInScen,MorningCIN,PlainsFcst,AMInsWliScen,LIfr12ZDENSd,AMDewptCalPl,N34StarFcst,LatestCIN,CurPropConv,LLIW


    AttrDict.make({
        "name": "CldShadeOth",
        "expression": "Other indicators of cloud shading",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CompPlFcst",
        "expression": "Composite plains area forecast",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "SatContMoist",
        "expression": "Satellite contribution to moisture",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CombMoisture",
        "expression": "Combined moisture levels",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "RaoContMoist",
        "expression": "Reading at the forecast center for moisture",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "VISCloudCov",
        "expression": "Visible cloud cover",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CombClouds",
        "expression": "Combined cloud cover",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "IRCloudCover",
        "expression": "Infrared cloud cover",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "InsInMt",
        "expression": "Instability in the mountains",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "AMInstabMt",
        "expression": "Atmospheric instability in the mountains",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "OutflowFrMt",
        "expression": "Outflow from mountains",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CldShadeConv",
        "expression": "Cloud shading in convective situations",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "MountainFcst",
        "expression": "Mountains (region 1) area weather forecast",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "WndHodograph",
        "expression": "Wind hodograph",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Boundaries",
        "expression": "Weather boundaries",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "MorningBound",
        "expression": "Morning weather boundaries",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "CapChange",
        "expression": "Change in the atmospheric cap or lid",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "InsChange",
        "expression": "Change in instability",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CapInScen",
        "expression": "Capping withing scenario",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "LoLevMoistAd",
        "expression": "Low-level moisture advection",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "InsSclInScen",
        "expression": "Scale of instability in scenarios",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "R5Fcst",
        "expression": "Region 5 forecast",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Date",
        "expression": "Date",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "Scenario",
        "expression": "Scenario",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ScenRelAMCIN",
        "expression": "Scenario relevant to AM convective inhibition",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ScenRelAMIns",
        "expression": "Scenario relevant to AM (morning) instability",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "ScenRel3_4",
        "expression": "Scenario relevant to regions 2/3/4",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ScnRelPlFcst",
        "expression": "Scenario relevant to plains forecast",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Dewpoints",
        "expression": "Dewpoints",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LowLLapse",
        "expression": "Low-level lapse rate",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "MeanRH",
        "expression": "Mean relative humidity",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "MidLLapse",
        "expression": "Mid-level lapse rate",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "MvmtFeatures",
        "expression": "Movement of features",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "RHRatio",
        "expression": "Relative humidity ratio",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "SfcWndShfDis",
        "expression": "Surface wind shifts and discontinuities",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "SynForcng",
        "expression": "Synoptic scale forcing",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "TempDis",
        "expression": "Temperature discontinuities",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "WindAloft",
        "expression": "Wind aloft",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "WindFieldMt",
        "expression": "Wind field in mountainous areas",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "WindFieldPln",
        "expression": "Wind field in plains areas",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "AMCINInScen",
        "expression": "AM convective inhibition in scenario",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "MorningCIN",
        "expression": "Morning convective inhibition",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "PlainsFcst",
        "expression": "Forecast for plains areas",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "AMInsWliScen",
        "expression": "AM instability within scenario",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "LIfr12ZDENSd",
        "expression": "Lifted Index from 12Z sounding",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
#LatestCIN,CurPropConv,LLIW
    AttrDict.make({
        "name": "AMDewptCalPl",
        "expression": "AM Dewpoint Calculated Plains",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "N34StarFcst",
        "expression": "Regions 2/3/4 forecast",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LatestCIN",
        "expression": "Latest Convective Inhibition",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "CurPropConv",
        "expression": "Current propensity to convection",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LLIW",
        "expression": "Low-Level inflow wind severe weather index",
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
    txt_file_path = f'Ranking/ranking_{dataset_name}.txt'
    csv_file_path = f'Ranking/ranking_{dataset_name}.csv'
    txt_to_csv(txt_file_path, csv_file_path)
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ['prompt'] + df.columns.tolist()[1:]
    num = variables.__len__()
    df_seventh = df.iloc[::num-1]
    df_seventh.to_csv(f'Ranking/ranking_{dataset_name}.csv', index=False)

if __name__ == "__main__":
    main()
