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
queries_path = f"{dataset_name}"

variables = [
    AttrDict.make({
        "name": "jordtype",
        "expression": "Soil Type",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "nmin",
        "expression": "Minimum Nitrogen",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "aar_mod",
        "expression": "Year Modified",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "potnmin",
        "expression": "Potential Minimum Nitrogen",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "exptgens",
        "expression": "Experimental Generations",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "rokap",
        "expression": "Organic Carbon Content",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "komm",
        "expression": "Municipality",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "nedbarea",
        "expression": "Precipitation Area",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "jordn",
        "expression": "Soil Nitrogen Content",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "mod_nmin",
        "expression": "Modified Minimum Nitrogen",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "forfrugt",
        "expression": "Previous Crop",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ngodnt",
        "expression": "Good Nitrogen Treatment",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ngodnn",
        "expression": "Bad Nitrogen Treatment",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "nprot",
        "expression": "Nitrogen Protein",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ntilg",
        "expression": "Nitrogen Addition",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "pesticid",
        "expression": "Pesticide Use",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "nopt",
        "expression": "Nitrogen Optimization",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "ngodn",
        "expression": "Good Nitrogen",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "ngtilg",
        "expression": "Good Nitrogen Addition",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "protein",
        "expression": "Protein Content",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "saatid",
        "expression": "Sowing Time",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "dgv1059",
        "expression": "Growth Days",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "dg25",
        "expression": "Growth Stage Days",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),




    AttrDict.make({
        "name": "frspdag",
        "expression": "Frost Days in Morning",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "dgv5980",
        "expression": "Alternative Growth Days",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "aks_m2",
        "expression": "Spikes per Square Meter",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "keraks",
        "expression": "Kernel Spikes",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "bgbyg",
        "expression": "Background Building",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "sort",
        "expression": "Variety",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "srtprot",
        "expression": "Variety Protein",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "sorttkv",
        "expression": "Variety Thousand Kernel Weight",
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
        "name": "srtsize",
        "expression": "Variety Size",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "nplac",
        "expression": "Nitrogen Place",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "aks_vgt",
        "expression": "Spike Weight",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "spndx",
        "expression": "Sowing Index",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "tkv",
        "expression": "Thousand Kernel Weight",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "saamng",
        "expression": "Sowing Amount",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "saakern",
        "expression": "Sowing Kernels",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "tkvs",
        "expression": "Thousand Kernel Weight Size",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "antplnt",
        "expression": "Number of Plants",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "partigerm",
        "expression": "Partial Germination",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "markgrm",
        "expression": "Mark Gram",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "jordinf",
        "expression": "Soil Infection",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "udb",
        "expression": "Database",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "slt22",
        "expression": "slt22",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "s2225",
        "expression": "s2225",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "s2528",
        "expression": "s2528",
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
