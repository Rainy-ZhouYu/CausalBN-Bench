import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'link'
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
    name_str = "N56_d_g,D0_56_d_p,N56_d_m,N56_d_f,Z_56_d_m,N55_d_m,N57_d_m,N58_d_m,N59_d_m,N55_d_g,N55_d_f,Z_56_d_f,N54_d_m,N57_d_f,N58_d_f,N59_d_f,N54_d_g,N54_d_f,Z_56_a_m,D0_56_a_m,N55_a_m,N57_a_m,N58_a_m,N59_a_m,D0_55_a_x,N55_a_f,Z_56_a_f,D0_56_a_f,N54_a_m,N57_a_f,N58_a_f,N59_a_f,D0_54_a_x,N54_a_f,N57_d_g,D0_57_d_p,Z_57_d_m,Z_57_d_f,D0_57_a_x,Z_57_a_m,Z_57_a_f,N58_d_g,D0_58_d_p,Z_58_d_m,Z_58_d_f,D0_58_a_x,Z_58_a_m,Z_58_a_f,N59_d_g,D0_59_d_p,Z_59_d_m,Z_59_d_f,D0_59_a_x,Z_59_a_m,Z_59_a_f,D0_54_d_p,Z_54_d_m,N23_d_m,N23_d_g,N23_d_f,Z_54_d_f,N25_d_m,N25_d_g,N25_d_f,Z_54_a_m,N23_a_m,D0_23_a_x,N23_a_f,Z_54_a_f,N25_a_m,D0_25_a_x,N25_a_f,D0_25_d_p,Z_25_d_m,N73_d_m,N73_d_g,N73_d_f,Z_25_d_f,N72_d_m,N72_d_g,N72_d_f,Z_25_a_m,N73_a_m,N73_a_f,Z_25_a_f,N72_a_m,N72_a_f,D0_73_d_p,D0_72_d_p,Z_72_d_m,N3_d_m,N7_d_m,N8_d_m,N3_d_g,N3_d_f,Z_72_d_f,N5_d_m,N7_d_f,N8_d_f,N5_d_g,N5_d_f,Z_72_a_m,N3_a_m,N7_a_m,N8_a_m,N3_a_f,Z_72_a_f,N5_a_m,N7_a_f,N8_a_f,N5_a_f,D0_55_d_p,Z_55_d_m,N26_d_m,N26_d_g,N26_d_f,Z_55_d_f,N24_d_m,N24_d_g,N24_d_f,Z_55_a_m,N26_a_m,D0_26_a_x,N26_a_f,Z_55_a_f,N24_a_m,D0_24_a_x,N24_a_f,D0_26_d_p,Z_26_d_m,N71_d_m,N71_d_g,N71_d_f,Z_26_d_f,N70_d_m,N70_d_g,N70_d_f,Z_26_a_m,N71_a_m,N71_a_f,Z_26_a_f,N70_a_m,N70_a_f,D0_71_d_p,Z_71_d_m,N67_d_m,N67_d_g,N67_d_f,Z_71_d_f,N66_d_m,N66_d_g,N66_d_f,Z_71_a_m,N67_a_m,N67_a_f,Z_71_a_f,N66_a_m,N66_a_f,D0_70_d_p,Z_70_d_m,N69_d_m,N69_d_g,N69_d_f,Z_70_d_f,N68_d_m,N68_d_g,N68_d_f,Z_70_a_m,N69_a_m,N69_a_f,Z_70_a_f,N68_a_m,N68_a_f,D0_69_d_p,D0_68_d_p,Z_68_d_m,N2_d_m,N4_d_m,N2_d_g,N2_d_f,Z_68_d_f,N1_d_m,N4_d_f,N1_d_g,N1_d_f,Z_68_a_m,N2_a_m,N4_a_m,N2_a_f,Z_68_a_f,N1_a_m,N4_a_f,N1_a_f,D0_67_d_p,Z_67_d_m,N65_d_m,N65_d_g,N65_d_f,Z_67_d_f,N64_d_m,N64_d_g,N64_d_f,Z_67_a_m,N65_a_m,N65_a_f,Z_67_a_f,N64_a_m,N64_a_f,D0_66_d_p,D0_65_d_p,Z_65_d_m,N63_d_m,N63_d_g,N63_d_f,Z_65_d_f,N62_d_m,N62_d_g,N62_d_f,Z_65_a_m,N63_a_m,N63_a_f,Z_65_a_f,N62_a_m,N62_a_f,D0_64_d_p,Z_64_d_m,N61_d_m,N61_d_g,N61_d_f,Z_64_d_f,N60_d_m,N60_d_g,N60_d_f,Z_64_a_m,N61_a_m,N61_a_f,Z_64_a_f,N60_a_m,N60_a_f,N27_d_g,D0_27_d_p,N27_d_m,N27_d_f,Z_27_d_m,N19_d_m,N28_d_m,N29_d_m,N30_d_m,N31_d_m,N32_d_m,N33_d_m,N34_d_m,N35_d_m,N36_d_m,N37_d_m,N38_d_m,N19_d_g,N19_d_f,Z_27_d_f,N17_d_m,N28_d_f,N29_d_f,N30_d_f,N31_d_f,N32_d_f,N33_d_f,N34_d_f,N35_d_f,N36_d_f,N37_d_f,N38_d_f,N17_d_g,N17_d_f,Z_27_a_m,D0_27_a_m,N19_a_m,D0_28_a_m,N29_a_m,N30_a_m,N31_a_m,N32_a_m,N33_a_m,N34_a_m,N35_a_m,N36_a_m,N37_a_m,N38_a_m,D0_19_a_x,N19_a_f,Z_27_a_f,D0_27_a_f,N17_a_m,D0_28_a_f,N29_a_f,N30_a_f,N31_a_f,N32_a_f,N33_a_f,N34_a_f,N35_a_f,N36_a_f,N37_a_f,N38_a_f,D0_17_a_x,N17_a_f,N28_d_g,D0_28_d_p,Z_28_d_m,Z_28_d_f,Z_28_a_m,Z_28_a_f,N29_d_g,D0_29_d_p,Z_29_d_m,Z_29_d_f,D0_29_a_x,Z_29_a_m,Z_29_a_f,N30_d_g,D0_30_d_p,Z_30_d_m,Z_30_d_f,D0_30_a_x,Z_30_a_m,Z_30_a_f,N31_d_g,D0_31_d_p,Z_31_d_m,Z_31_d_f,D0_31_a_x,Z_31_a_m,Z_31_a_f,N32_d_g,D0_32_d_p,Z_32_d_m,Z_32_d_f,D0_32_a_x,Z_32_a_m,Z_32_a_f,N33_d_g,D0_33_d_p,Z_33_d_m,Z_33_d_f,D0_33_a_x,Z_33_a_m,Z_33_a_f,N34_d_g,D0_34_d_p,Z_34_d_m,Z_34_d_f,D0_34_a_x,Z_34_a_m,Z_34_a_f,N35_d_g,D0_35_d_p,Z_35_d_m,Z_35_d_f,D0_35_a_x,Z_35_a_m,Z_35_a_f,N36_d_g,D0_36_d_p,Z_36_d_m,Z_36_d_f,D0_36_a_x,Z_36_a_m,Z_36_a_f,N37_d_g,D0_37_d_p,Z_37_d_m,Z_37_d_f,D0_37_a_x,Z_37_a_m,Z_37_a_f,N38_d_g,D0_38_d_p,Z_38_d_m,Z_38_d_f,D0_38_a_x,Z_38_a_m,Z_38_a_f,D0_17_d_p,Z_17_d_m,N18_d_m,N7_d_g,Z_17_d_f,N12_d_m,N18_d_f,N12_d_g,N12_d_f,Z_17_a_m,N18_a_m,Z_17_a_f,N12_a_m,N18_a_f,N12_a_f,N39_d_g,D0_39_d_p,N39_d_m,N39_d_f,Z_39_d_m,N20_d_m,N40_d_m,N20_d_g,N20_d_f,Z_39_d_f,N40_d_f,N18_d_g,Z_39_a_m,D0_39_a_m,N20_a_m,N40_a_m,D0_20_a_x,N20_a_f,Z_39_a_f,D0_39_a_f,N40_a_f,D0_18_a_x,N40_d_g,D0_40_d_p,Z_40_d_m,Z_40_d_f,D0_40_a_x,Z_40_a_m,Z_40_a_f,D0_18_d_p,Z_18_d_m,Z_18_d_f,Z_18_a_m,Z_18_a_f,D0_7_d_p,Z_7_d_m,Z_7_d_f,Z_7_a_m,Z_7_a_f,N41_d_g,D0_41_d_p,N41_d_m,N41_d_f,Z_41_d_m,N21_d_m,N42_d_m,N43_d_m,N44_d_m,N45_d_m,N46_d_m,N47_d_m,N48_d_m,N49_d_m,N50_d_m,N51_d_m,N52_d_m,N53_d_m,N21_d_g,N21_d_f,Z_41_d_f,N22_d_m,N42_d_f,N43_d_f,N44_d_f,N45_d_f,N46_d_f,N47_d_f,N48_d_f,N49_d_f,N50_d_f,N51_d_f,N52_d_f,N53_d_f,N22_d_g,N22_d_f,Z_41_a_m,D0_41_a_m,N21_a_m,N42_a_m,N43_a_m,N44_a_m,N45_a_m,N46_a_m,N47_a_m,N48_a_m,N49_a_m,N50_a_m,N51_a_m,N52_a_m,N53_a_m,D0_21_a_x,N21_a_f,Z_41_a_f,D0_41_a_f,N22_a_m,N42_a_f,N43_a_f,N44_a_f,N45_a_f,N46_a_f,N47_a_f,N48_a_f,N49_a_f,N50_a_f,N51_a_f,N52_a_f,N53_a_f,D0_22_a_x,N22_a_f,N42_d_g,D0_42_d_p,Z_42_d_m,Z_42_d_f,D0_42_a_x,Z_42_a_m,Z_42_a_f,N43_d_g,D0_43_d_p,Z_43_d_m,Z_43_d_f,D0_43_a_x,Z_43_a_m,Z_43_a_f,N44_d_g,D0_44_d_p,Z_44_d_m,Z_44_d_f,D0_44_a_x,Z_44_a_m,Z_44_a_f,N45_d_g,D0_45_d_p,Z_45_d_m,Z_45_d_f,D0_45_a_x,Z_45_a_m,Z_45_a_f,N46_d_g,D0_46_d_p,Z_46_d_m,Z_46_d_f,D0_46_a_x,Z_46_a_m,Z_46_a_f,N47_d_g,D0_47_d_p,Z_47_d_m,Z_47_d_f,D0_47_a_x,Z_47_a_m,Z_47_a_f,N48_d_g,D0_48_d_p,Z_48_d_m,Z_48_d_f,D0_48_a_x,Z_48_a_m,Z_48_a_f,N49_d_g,D0_49_d_p,Z_49_d_m,Z_49_d_f,D0_49_a_x,Z_49_a_m,Z_49_a_f,N50_d_g,D0_50_d_p,Z_50_d_m,Z_50_d_f,D0_50_a_x,Z_50_a_m,Z_50_a_f,N51_d_g,D0_51_d_p,Z_51_d_m,Z_51_d_f,D0_51_a_x,Z_51_a_m,Z_51_a_f,N52_d_g,D0_52_d_p,Z_52_d_m,Z_52_d_f,D0_52_a_x,Z_52_a_m,Z_52_a_f,N53_d_g,D0_53_d_p,Z_53_d_m,Z_53_d_f,D0_53_a_x,Z_53_a_m,Z_53_a_f,D0_22_d_p,Z_22_d_m,N8_d_g,Z_22_d_f,N14_d_m,N14_d_g,N14_d_f,Z_22_a_m,Z_22_a_f,N14_a_m,N14_a_f,D0_8_d_p,Z_8_d_m,Z_8_d_f,Z_8_a_m,Z_8_a_f,D0_3_d_p,Z_3_d_m,Z_3_d_f,Z_3_a_m,Z_3_a_f,D0_19_d_p,Z_19_d_m,N13_d_m,N13_d_g,N13_d_f,Z_19_d_f,N9_d_m,N9_d_g,N9_d_f,Z_19_a_m,N13_a_m,D0_13_a_x,N13_a_f,Z_19_a_f,N9_a_m,D0_9_a_x,N9_a_f,D0_20_d_p,Z_20_d_m,Z_20_d_f,Z_20_a_m,Z_20_a_f,D0_21_d_p,Z_21_d_m,Z_21_d_f,Z_21_a_m,Z_21_a_f,D0_9_d_p,Z_9_d_m,N6_d_m,N10_d_m,N11_d_m,N6_d_g,N6_d_f,Z_9_d_f,N10_d_f,N11_d_f,N4_d_g,Z_9_a_m,N6_a_m,N10_a_m,N11_a_m,N6_a_f,Z_9_a_f,N10_a_f,N11_a_f,D0_23_d_p,Z_23_d_m,N15_d_m,N15_d_g,N15_d_f,Z_23_d_f,N10_d_g,Z_23_a_m,N15_a_m,N15_a_f,Z_23_a_f,D0_10_d_p,Z_10_d_m,Z_10_d_f,Z_10_a_m,Z_10_a_f,D0_24_d_p,Z_24_d_m,N16_d_m,N16_d_g,N16_d_f,Z_24_d_f,N11_d_g,Z_24_a_m,N16_a_m,N16_a_f,Z_24_a_f,D0_11_d_p,Z_11_d_m,Z_11_d_f,Z_11_a_m,Z_11_a_f,D0_4_d_p,Z_4_d_m,Z_4_d_f,Z_4_a_m,Z_4_a_f,D0_2_d_p,Z_2_d_m,Z_2_d_f,Z_2_a_m,Z_2_a_f,D0_63_d_p,D0_62_d_p,D0_1_d_p,Z_1_d_m,Z_1_d_f,Z_1_a_m,Z_1_a_f,D0_61_d_p,D0_60_d_p,D0_16_d_p,D0_15_d_p,D0_14_d_p,D0_13_d_p,D0_12_d_p,D0_6_d_p,D0_5_d_p"

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
