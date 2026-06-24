import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'barley'
queries_path = f"./question/{dataset_name}"
# jordtype,nmin,aar_mod,potnmin,exptgens,rokap,komm,nedbarea,jordn,mod_nmin,forfrugt,ngodnt,ngodnn,nprot,ntilg,pesticid,nopt,ngodn,ngtilg,protein,saatid,dgv1059,dg25,frspdag,dgv5980,aks_m2,keraks,bgbyg,sort,srtprot,sorttkv,srtsize,nplac,aks_vgt,spndx,tkv,saamng,saakern,tkvs,antplnt,partigerm,markgrm,jordinf,udb,ksort,slt22,s2225,s2528
# jordtype (Soil Type): 土壤类型。
# nmin (Minimum Nitrogen): 最小氮含量。
# aar_mod (Year Modified): 修改年份。
# potnmin (Potential Minimum Nitrogen): 潜在的最小氮含量。
# exptgens (Experimental Generations): 实验代数。
# rokap (Organic Carbon Content): 有机碳含量。
# komm (Municipality): 地方自治体或区域。
# nedbarea (Precipitation Area): 降水区域。
# jordn (Soil Nitrogen Content): 土壤氮含量。
# mod_nmin (Modified Minimum Nitrogen): 修改后的最小氮含量。
# forfrugt (Previous Crop): 前作物。
# ngodnt (Good Nitrogen Treatment): 良好的氮处理。
# ngodnn (Bad Nitrogen Treatment): 较差的氮处理。
# nprot (Nitrogen Protein): 氮蛋白。
# ntilg (Nitrogen Addition): 氮添加。
# pesticid (Pesticide Use): 农药使用。
# nopt (Nitrogen Optimization): 氮最优化。
# ngodn (Good Nitrogen): 良好的氮。
# ngtilg (Good Nitrogen Addition): 良好的氮添加。
# protein (Protein Content): 蛋白质含量。
# saatid (Sowing Time): 播种时间。
# dgv1059 (Growth Days): 生长天数。
# dg25 (Growth Stage Days): 生长阶段天数。
# frspdag (Frost Days in Morning): 早晨霜冻天数。
# dgv5980 (Alternative Growth Days): 另一种生长天数计算。
# aks_m2 (Spikes per Square Meter): 每平方米穗数。
# keraks (Kernel Spikes): 穗粒。
# bgbyg (Background Building): 背景建筑。
# sort (Variety): 品种。
# srtprot (Variety Protein): 品种蛋白。
# sorttkv (Variety Thousand Kernel Weight): 品种千粒重。
# srtsize (Variety Size): 品种大小。
# nplac (Nitrogen Place): 氮位置。
# aks_vgt (Spike Weight): 穗重。
# spndx (Sowing Index): 播种指数。
# tkv (Thousand Kernel Weight): 千粒重。
# saamng (Sowing Amount): 播种量。
# saakern (Sowing Kernels): 播种粒数。
# tkvs (Thousand Kernel Weight Size): 千粒重大小。
# antplnt (Number of Plants): 植物数量。
# partigerm (Partial Germination): 部分发芽。
# markgrm (Mark Gram): 标记克拉姆。
# jordinf (Soil Infection): 土壤感染。
# udb (Database): 数据库。
# ksort (Nitrogen Sorting): 氮排序。
# slt22, s2225, s2528: 可能是与特定土壤或气候条件相关的指标。


def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    name_str = "jordtype,nmin,aar_mod,potnmin,exptgens,rokap,komm,nedbarea,jordn,mod_nmin,forfrugt,ngodnt,ngodnn,nprot,ntilg,pesticid,nopt,ngodn,ngtilg,protein,saatid,dgv1059,dg25,frspdag,dgv5980,aks_m2,keraks,bgbyg,sort,srtprot,sorttkv,srtsize,nplac,aks_vgt,spndx,tkv,saamng,saakern,tkvs,antplnt,partigerm,markgrm,jordinf,udb,ksort,slt22,s2225,s2528"
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
    txt_file_path = f'question/questions_{dataset_name}.txt'
    csv_file_path = f'question/questions_{dataset_name}.csv'
    txt_to_csv(txt_file_path, csv_file_path)
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ['prompt'] + df.columns.tolist()[1:]

    # 将DataFrame保存到新的csv中
    df.to_csv(f'question/questions_{dataset_name}.csv', index=False)

if __name__ == "__main__":
    main()
