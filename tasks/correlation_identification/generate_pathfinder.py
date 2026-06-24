import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'pathfinder'
queries_path = f"./question/{dataset_name}"
# GOAL_2,GOAL_48,SNode_3,RApp1,SNode_47,SNode_4,SNode_75,SNode_123,SNode_155,SNode_5,GOAL_49,SNode_6,SNode_7,SNode_59,DISPLACEM0,SNode_8,GIVEN_1,RApp2,SNode_42,SNode_9,SNode_41,SNode_10,SNode_33,SNode_34,SNode_11,SNode_26,RApp4,SNode_12,SNode_91,SNode_92,SNode_13,SNode_15,SNode_25,SNode_16,SNode_20,SNode_94,SNode_17,SNode_51,NEED1,SNode_21,GOAL_150,GRAV2,SNode_24,VALUE3,SNode_31,SNode_40,SNode_46,SLIDING4,SNode_88,CONSTANT5,RApp3,KNOWN6,VELOCITY7,SNode_27,KNOWN8,SNode_28,COMPO16,RApp6,RApp12,TRY12,TRY11,TRY13,TRY14,TRY15,GOAL_50,GOAL_53,GOAL_56,CHOOSE19,SYSTEM18,SNode_52,KINEMATI17,IDENTIFY10,SNode_43,SNode_54,SNode_55,IDENTIFY9,SNode_29,GOAL_63,VAR20,SNode_38,SNode_44,SNode_65,GIVEN21,VECTOR27,RApp5,RApp11,RApp13,APPLY32,GOAL_57,GOAL_61,CHOOSE35,MAXIMIZE34,SNode_106,SNode_60,AXIS33,SNode_152,WRITE31,GOAL_62,WRITE30,GOAL_66,GOAL_69,GOAL_72,SNode_74,RESOLVE37,SNode_64,NEED36,SNode_67,SNode_70,SNode_73,RApp8,SNode_115,SNode_117,SNode_122,RApp10,SNode_131,SNode_133,SNode_154,IDENTIFY39,RESOLVE38,SNode_68,IDENTIFY41,RESOLVE40,SNode_71,IDENTIFY43,RESOLVE42,KINE29,GOAL_79,VECTOR44,GOAL_80,EQUATION28,GOAL_142,GOAL_143,GOAL_146,GOAL_81,GOAL_83,TRY25,TRY24,TRY26,GOAL_84,GOAL_98,GOAL_87,GOAL_103,CHOOSE47,SNode_86,SYSTEM46,SNode_156,NEWTONS45,DEFINE23,SNode_37,GOAL_99,SNode_100,IDENTIFY22,SNode_124,SNode_136,NULL48,FIND49,SNode_93,SNode_97,SNode_102,NORMAL50,STRAT_90,NORMAL52,INCLINE51,HORIZ53,BUGGY54,SNode_118,SNode_134,SNode_120,SNode_135,IDENTIFY55,SNode_119,WEIGHT56,SNode_95,WEIGHT57,SNode_116,SNode_132,FIND58,IDENTIFY59,FORCE60,APPLY61,GOAL_104,GOAL_107,CHOOSE62,WRITE63,GOAL_108,GOAL_109,WRITE64,GOAL_126,GOAL_127,RApp9,SNode_137,GOAL_153,GOAL_110,GOAL_111,RApp7,GOAL_121,SNode_125,GOAL65,GOAL_113,GOAL_114,GOAL66,NEED67,SNode_112,GOAL68,GOAL_129,GOAL_130,VECTOR69,VECTOR70,EQUAL71,GOAL72,VECTOR73,NEWTONS74,SUM75,SNode_128,GOAL_147,GOAL_149,TRY76,APPLY77,SNode_151,GRAV78
# Assuming AttrDict is a custom class similar to a dictionary, but with attribute-style access.
# Since AttrDict is not a built-in type, I will define a simple mock-up for demonstration.
# Original string of variable names

# Creating a list of AttrDict objects


def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    name_str = "Fault,F1,F97,F2,F78,F3,F4,F5,F53,F6,F7,F56,F8,F9,F10,F55,F52,F11,F12,F13,F14,F15,F16,F17,F18,F19,F41,F44,F20,F90,F21,F22,F23,F24,F25,F26,F27,F28,F92,F98,F30,F31,F32,F33,F34,F35,F36,F37,F84,F96,F38,F39,F40,F42,F43,F45,F46,F47,F85,F48,F49,F50,F51,F83,F54,F57,F58,F59,F60,F61,F62,F63,F64,F65,F66,F67,F68,F69,F72,F70,F71,F73,F74,F75,F76,F77,F79,F80,F81,F82,F87,F88,F89,F91,F93,F94,F95,F99,F100,F105,F101,F102,F103,F104,F106,F107,F108,F86,F29"
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
