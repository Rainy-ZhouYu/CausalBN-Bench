import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'hepar2'
queries_path = f"./Ranking/{dataset_name}"
# alcoholism,THepatitis,Steatosis,vh_amn,ChHepatitis,hbsag,hbsag_anti,hbc_anti,hcv_anti,hbeag,hepatotoxic,RHepatitis,fatigue,phosphatase,inr,hepatomegaly,alt,ast,ggtp,anorexia,nausea,spleen,hospital,injections,transfusion,surgery,gallstones,choledocholithotomy,bilirubin,upper_pain,fat,pressure_ruq,flatulence,amylase,fibrosis,ESR,cholesterol,sex,PBC,Hyperbilirubinemia,age,ama,le_cells,joints,pain,platelet,encephalopathy,carcinoma,Cirrhosis,diabetes,obesity,triglycerides,pain_ruq,proteins,edema,alcohol,spiders,albumin,edge,irregular_liver,palms,itching,skin,jaundice,ascites,bleeding,urea,density,consciousness,hepatalgia# N0_7muVerMo: 可能是指在 0 到 7 µm 高度范围内的垂直水汽含量。
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
        "name": "alcoholism",
        "expression": "alcoholism",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "THepatitis",
        "expression": "Toxic Hepatitis",
        "singular": False,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Steatosis",
        "expression": "Steatosis",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
# CldShadeOth,CompPlFcst,SatContMoist,CombMoisture,RaoContMoist,VISCloudCov,CombClouds,IRCloudCover,InsInMt,AMInstabMt,OutflowFrMt,CldShadeConv,MountainFcst,WndHodograph,Boundaries,MorningBound,CapChange,InsChange,CapInScen,LoLevMoistAd,InsSclInScen,R5Fcst,Date,Scenario,ScenRelAMCIN,ScenRelAMIns,ScenRel3_4,ScnRelPlFcst,Dewpoints,LowLLapse,MeanRH,MidLLapse,MvmtFeatures,RHRatio,SfcWndShfDis,SynForcng,TempDis,WindAloft,WindFieldMt,WindFieldPln,AMCINInScen,MorningCIN,PlainsFcst,AMInsWliScen,LIfr12ZDENSd,AMDewptCalPl,N34StarFcst,LatestCIN,CurPropConv,LLIW

    AttrDict.make({
        "name": "vh_amn",
        "expression": "vh_amn",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ChHepatitis",
        "expression": "Chronic Hepatitis",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "hbsag",
        "expression": "Hepatitis B surface antigen",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
# CldShadeOth,CompPlFcst,SatContMoist,CombMoisture,RaoContMoist,VISCloudCov,CombClouds,IRCloudCover,InsInMt,AMInstabMt,OutflowFrMt,CldShadeConv,MountainFcst,WndHodograph,Boundaries,MorningBound,CapChange,InsChange,CapInScen,LoLevMoistAd,InsSclInScen,R5Fcst,Date,Scenario,ScenRelAMCIN,ScenRelAMIns,ScenRel3_4,ScnRelPlFcst,Dewpoints,LowLLapse,MeanRH,MidLLapse,MvmtFeatures,RHRatio,SfcWndShfDis,SynForcng,TempDis,WindAloft,WindFieldMt,WindFieldPln,AMCINInScen,MorningCIN,PlainsFcst,AMInsWliScen,LIfr12ZDENSd,AMDewptCalPl,N34StarFcst,LatestCIN,CurPropConv,LLIW


    AttrDict.make({
        "name": "hbsag_anti",
        "expression": "Hepatitis B surface antibody",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "hbc_anti",
        "expression": "Hepatitis B core antibody",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "hcv_anti",
        "expression": "Hepatitis C antibody",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "hbeag",
        "expression": "Hepatitis B e antigen",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "hepatotoxic",
        "expression": "hepatotoxic",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "RHepatitis",
        "expression": "Reactive Hepatitis",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "fatigue",
        "expression": "fatigue",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "phosphatase",
        "expression": "phosphatase",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "inr",
        "expression": "International Normalized Ratio",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "hepatomegaly",
        "expression": "hepatomegaly",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "alt",
        "expression": "Alanine aminotransferase",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ast",
        "expression": "Aspartate aminotransferase",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "ggtp",
        "expression": "Gamma-glutamyl transferase",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "anorexia",
        "expression": "anorexia",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "nausea",
        "expression": "nausea",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "spleen",
        "expression": "spleen",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "hospital",
        "expression": "hospital",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "injections",
        "expression": "injections",
        "singular": False,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "transfusion",
        "expression": "transfusion",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "surgery",
        "expression": "surgery",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "gallstones",
        "expression": "gallstones",
        "singular": False,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "choledocholithotomy",
        "expression": "choledocholithotomy",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "bilirubin",
        "expression": "bilirubin",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "upper_pain",
        "expression": "upper pain",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "fat",
        "expression": "fat",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "pressure_ruq",
        "expression": "Right Upper Quadrant pressure",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "flatulence",
        "expression": "flatulence",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "amylase",
        "expression": "amylase",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "fibrosis",
        "expression": "fibrosis",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ESR",
        "expression": "Erythrocyte Sedimentation Rate",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "cholesterol",
        "expression": "cholesterol",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "sex",
        "expression": "sex",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PBC",
        "expression": "Primary Biliary Cirrhosis",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Hyperbilirubinemia",
        "expression": "Hyperbilirubinemia",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "age",
        "expression": "age",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ama",
        "expression": "Anti-mitochondrial antibodies",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "le_cells",
        "expression": "LE cells",
        "singular": False,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "joints",
        "expression": "joints",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "pain",
        "expression": "pain",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "platelet",
        "expression": "platelet",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "encephalopathy",
        "expression": "encephalopathy",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "carcinoma",
        "expression": "carcinoma",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "Cirrhosis",
        "expression": "Cirrhosis",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "diabetes",
        "expression": "diabetes",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "obesity",
        "expression": "obesity",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "triglycerides",
        "expression": "triglycerides",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "pain_ruq",
        "expression": "Pain Right Upper Quadrant",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
#LatestCIN,CurPropConv,LLIW
    AttrDict.make({
        "name": "proteins",
        "expression": "proteins",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "edema",
        "expression": "edema",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "alcohol",
        "expression": "alcohol",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "spiders",
        "expression": "Spider Angiomas",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "albumin",
        "expression": "albumin",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

   AttrDict.make({
        "name": "edge",
        "expression": "edge",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "irregular_liver",
        "expression": "irregular liver",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
#LatestCIN,CurPropConv,LLIW
    AttrDict.make({
        "name": "palms",
        "expression": "Palmar Erythema",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "itching",
        "expression": "itching",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "skin",
        "expression": "skin",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "jaundice",
        "expression": "jaundice",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ascites",
        "expression": "ascites",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "bleeding",
        "expression": "bleeding",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "urea",
        "expression": "urea",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "density",
        "expression": "density",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "consciousness",
        "expression": "consciousness",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "hepatalgia",
        "expression": "Hepatic Pain",
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
