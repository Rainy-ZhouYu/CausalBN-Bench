import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
queries_path = "./queries/alarm"
# MINVOLSET
# DISCONNECT
# VENTMACH
# PULMEMBOLUS
# PAP
# SHUNT
# INTUBATION
# KINKEDTUBE
# VENTTUBE
# PRESS
# VENTLUNG
# HISTORY
# LVFAILURE
# LVEDVOLUME
# PCWP
# CVP
# ANAPHYLAXIS
# FI02
# TPR
# INSUFFANESTH
# HYPOVOLEMIA
# STROKEVOLUME
# ERRLOWOUTPUT
# co
# HRBP
# MINVOL
# PVSAT
# SAO2
# CATECHOL
# HR
# HRSAT
# VENTALV
# ARTCO2
# EXPCO2
# ERRCAUTER
# HREKG
# BP


variables = [
    AttrDict.make({
        "name": "MINVOLSET",
        "expression": "MINVOLSET",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "DISCONNECT",
        "expression": "DISCONNECT",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "VENTMACH",
        "expression": "VENTMACH",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PULMEMBOLUS",
        "expression": "PULMEMBOLUS",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PAP",
        "expression": "PAP",
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
    }),    AttrDict.make({
        "name": "KINKEDTUBE",
        "expression": "KINKEDTUBE",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "VENTTUBE",
        "expression": "VENTTUBE",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "PRESS",
        "expression": "PRESS",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "VENTLUNG",
        "expression": "VENTLUNG",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "HISTORY",
        "expression": "HISTORY",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "LVFAILURE",
        "expression": "LVFAILURE",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "LVEDVOLUME",
        "expression": "LVEDVOLUME",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "PCWP",
        "expression": "PCWP",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "CVP",
        "expression": "CVP",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "ANAPHYLAXIS",
        "expression": "ANAPHYLAXIS",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "FI02",
        "expression": "FI02",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "TPR",
        "expression": "TPR",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "INSUFFANESTH",
        "expression": "INSUFFANESTH",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "HYPOVOLEMIA",
        "expression": "HYPOVOLEMIA",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "STROKEVOLUME",
        "expression": "STROKEVOLUME",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "ERRLOWOUTPUT",
        "expression": "ERRLOWOUTPUT",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),




    AttrDict.make({
        "name": "co",
        "expression": "co",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "HRBP",
        "expression": "HRBP",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "MINVOL",
        "expression": "MINVOL",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "PVSAT",
        "expression": "PVSAT",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "SAO2",
        "expression": "SAO2",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "CATECHOL",
        "expression": "CATECHOL",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "HR",
        "expression": "HR",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "HRSAT",
        "expression": "HRSAT",
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
        "name": "VENTALV",
        "expression": "VENTALV",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "ARTCO2",
        "expression": "ARTCO2",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "EXPCO2",
        "expression": "EXPCO2",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),    AttrDict.make({
        "name": "ERRCAUTER",
        "expression": "ERRCAUTER",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "HREKG",
        "expression": "HREKG",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
AttrDict.make({
        "name": "BP",
        "expression": "BP",
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
    txt_file_path = 'queries/questions_alarm.txt'
    csv_file_path = 'queries/questions_alarm.csv'
    txt_to_csv(txt_file_path, csv_file_path)

if __name__ == "__main__":
    main()
