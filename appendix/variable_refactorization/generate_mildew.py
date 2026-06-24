import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'mildew'
queries_path = f"./question/{dataset_name}"
# dm_1,dm_2,foto_1,straaling_1,temp_1,mikro_1,lai_1,lai_2,meldug_1,meldug_2,lai_0,dm_3,foto_2,straaling_2,temp_2,mikro_2,lai_3,meldug_3,dm_4,foto_3,straaling_3,temp_3,mikro_3,lai_4,meldug_4,udbytte,foto_4,straaling_4,temp_4,middel_1,middel_2,middel_3,nedboer_1,nedboer_2,nedboer_3

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    name_str = "dm_1,dm_2,foto_1,straaling_1,temp_1,mikro_1,lai_1,lai_2,meldug_1,meldug_2,lai_0,dm_3,foto_2,straaling_2,temp_2,mikro_2,lai_3,meldug_3,dm_4,foto_3,straaling_3,temp_3,mikro_3,lai_4,meldug_4,udbytte,foto_4,straaling_4,temp_4,middel_1,middel_2,middel_3,nedboer_1,nedboer_2,nedboer_3"    # Splitting the string into a list and formatting each element
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
