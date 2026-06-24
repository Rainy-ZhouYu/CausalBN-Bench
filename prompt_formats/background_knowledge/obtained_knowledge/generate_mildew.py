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
queries_path = f"{dataset_name}"

variables = [
    AttrDict.make({
        "name": "dm_1",
        "expression": "Dry Matter at the first time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "dm_2",
        "expression": "Dry Matter at the second time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "foto_1",
        "expression": "Photosynthesis at the first time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "straaling_1",
        "expression": "Radiation at the first time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "temp_1",
        "expression": "Temperature at the first time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    #mikro_1,lai_1,lai_2,meldug_1,meldug_2,lai_0,dm_3,foto_2,straaling_2,temp_2,mikro_2,lai_3,meldug_3,dm_4,foto_3,straaling_3,temp_3,mikro_3,lai_4,meldug_4,udbytte,foto_4,straaling_4,temp_4,middel_1,middel_2,middel_3,nedboer_1,nedboer_2,nedboer_3
    AttrDict.make({
        "name": "mikro_1",
        "expression": "Microorganism at the first time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "lai_1",
        "expression": "Leaf area index at the first time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "lai_2",
        "expression": "Leaf area index at the second time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    #dm_3,foto_2,straaling_2,temp_2,mikro_2,lai_3,meldug_3,dm_4,foto_3,straaling_3,temp_3,mikro_3,lai_4,meldug_4,udbytte,foto_4,straaling_4,temp_4,middel_1,middel_2,middel_3,nedboer_1,nedboer_2,nedboer_3

    AttrDict.make({
        "name": "meldug_1",
        "expression": "Mildew at the first time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "meldug_2",
        "expression": "Mildew at the second time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "lai_0",
        "expression": "Leaf area index at the zero time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
#temp_2,mikro_2,lai_3,meldug_3,dm_4,foto_3,straaling_3,temp_3,mikro_3,lai_4,meldug_4,udbytte,foto_4,straaling_4,temp_4,middel_1,middel_2,middel_3,nedboer_1,nedboer_2,nedboer_3

    AttrDict.make({
        "name": "dm_3",
        "expression": "Dry Matter at the three time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "foto_2",
        "expression": "Photosynthesis at the second time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "straaling_2",
        "expression": "Radiation at the second time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
#dm_4,foto_3,straaling_3,temp_3,mikro_3,lai_4,meldug_4,udbytte,foto_4,straaling_4,temp_4,middel_1,middel_2,middel_3,nedboer_1,nedboer_2,nedboer_3


    AttrDict.make({
        "name": "temp_2",
        "expression": "Temperature at the second time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "mikro_2",
        "expression": "Microorganism at the second time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "lai_3",
        "expression": "Leaf area index at the third time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "meldug_3",
        "expression": "Mildew at the third time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

#lai_4,meldug_4,udbytte,foto_4,straaling_4,temp_4,middel_1,middel_2,middel_3,nedboer_1,nedboer_2,nedboer_3

    AttrDict.make({
        "name": "dm_4",
        "expression": "Dry Matter at the fourth time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "foto_3",
        "expression": "Photosynthesis at the third time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "straaling_3",
        "expression": "Radiation at the third time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "temp_3",
        "expression": "Temperature at the third time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "mikro_3",
        "expression": "Microorganism at the third time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "lai_4",
        "expression": "Leaf Area Index at the fourth time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "meldug_4",
        "expression": "Mildew at the fourth time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "udbytte",
        "expression": "udbytte",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "foto_4",
        "expression": "Photosynthesis at the fourth time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "straaling_4",
        "expression": "Radiation at the fourth time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "temp_4",
        "expression": "Temperature at the fourth time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "middel_1",
        "expression": "Treatment method 1 or Pesticide 1",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "middel_2",
        "expression": "Treatment method 2 or Pesticide 2",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "middel_3",
        "expression": "Treatment method 3 or Pesticide 3",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "nedboer_1",
        "expression": "Precipitation at the first time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "nedboer_2",
        "expression": "Precipitation at the second time stage",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "nedboer_3",
        "expression": "Precipitation at the third time stage",
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
