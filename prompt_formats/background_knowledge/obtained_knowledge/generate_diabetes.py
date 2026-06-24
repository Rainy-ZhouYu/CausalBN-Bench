import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'diabetes'
queries_path = f"{dataset_name}"

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    name_str = "cho_init,cho_0,ins_sens,activ_ins_0,activ_ins_1,activ_ins_2,activ_ins_3,activ_ins_4,activ_ins_5,activ_ins_6,activ_ins_7,activ_ins_8,activ_ins_9,activ_ins_10,activ_ins_11,activ_ins_12,activ_ins_13,activ_ins_14,activ_ins_15,activ_ins_16,activ_ins_17,activ_ins_18,activ_ins_19,activ_ins_20,activ_ins_21,activ_ins_22,activ_ins_23,meal_0,gut_abs_0,cho_bal_0,bg_0,renal_cl_0,ins_indep_util_0,ins_dep_util_0,glu_prod_0,bg_1,ins_abs_0,basal_bal_0,ins_indep_0,ins_dep_0,endo_bal_0,cho_1,tot_bal_0,met_irr_0,meal_1,gut_abs_1,cho_bal_1,renal_cl_1,ins_indep_util_1,ins_dep_util_1,glu_prod_1,bg_2,ins_abs_1,basal_bal_1,ins_indep_1,ins_dep_1,endo_bal_1,cho_2,tot_bal_1,met_irr_1,meal_2,gut_abs_2,cho_bal_2,renal_cl_2,ins_indep_util_2,ins_dep_util_2,glu_prod_2,bg_3,ins_abs_2,basal_bal_2,ins_indep_2,ins_dep_2,endo_bal_2,cho_3,tot_bal_2,met_irr_2,meal_3,gut_abs_3,cho_bal_3,renal_cl_3,ins_indep_util_3,ins_dep_util_3,glu_prod_3,bg_4,ins_abs_3,basal_bal_3,ins_indep_3,ins_dep_3,endo_bal_3,cho_4,tot_bal_3,met_irr_3,meal_4,gut_abs_4,cho_bal_4,renal_cl_4,ins_indep_util_4,ins_dep_util_4,glu_prod_4,bg_5,ins_abs_4,basal_bal_4,ins_indep_4,ins_dep_4,endo_bal_4,cho_5,tot_bal_4,met_irr_4,meal_5,gut_abs_5,cho_bal_5,renal_cl_5,ins_indep_util_5,ins_dep_util_5,glu_prod_5,bg_6,ins_abs_5,basal_bal_5,ins_indep_5,ins_dep_5,endo_bal_5,cho_6,tot_bal_5,met_irr_5,meal_6,gut_abs_6,cho_bal_6,renal_cl_6,ins_indep_util_6,ins_dep_util_6,glu_prod_6,bg_7,ins_abs_6,basal_bal_6,ins_indep_6,ins_dep_6,endo_bal_6,cho_7,tot_bal_6,met_irr_6,meal_7,gut_abs_7,cho_bal_7,renal_cl_7,ins_indep_util_7,ins_dep_util_7,glu_prod_7,bg_8,ins_abs_7,basal_bal_7,ins_indep_7,ins_dep_7,endo_bal_7,cho_8,tot_bal_7,met_irr_7,meal_8,gut_abs_8,cho_bal_8,renal_cl_8,ins_indep_util_8,ins_dep_util_8,glu_prod_8,bg_9,ins_abs_8,basal_bal_8,ins_indep_8,ins_dep_8,endo_bal_8,cho_9,tot_bal_8,met_irr_8,meal_9,gut_abs_9,cho_bal_9,renal_cl_9,ins_indep_util_9,ins_dep_util_9,glu_prod_9,bg_10,ins_abs_9,basal_bal_9,ins_indep_9,ins_dep_9,endo_bal_9,cho_10,tot_bal_9,met_irr_9,meal_10,gut_abs_10,cho_bal_10,renal_cl_10,ins_indep_util_10,ins_dep_util_10,glu_prod_10,bg_11,ins_abs_10,basal_bal_10,ins_indep_10,ins_dep_10,endo_bal_10,cho_11,tot_bal_10,met_irr_10,meal_11,gut_abs_11,cho_bal_11,renal_cl_11,ins_indep_util_11,ins_dep_util_11,glu_prod_11,bg_12,ins_abs_11,basal_bal_11,ins_indep_11,ins_dep_11,endo_bal_11,cho_12,tot_bal_11,met_irr_11,meal_12,gut_abs_12,cho_bal_12,renal_cl_12,ins_indep_util_12,ins_dep_util_12,glu_prod_12,bg_13,ins_abs_12,basal_bal_12,ins_indep_12,ins_dep_12,endo_bal_12,cho_13,tot_bal_12,met_irr_12,meal_13,gut_abs_13,cho_bal_13,renal_cl_13,ins_indep_util_13,ins_dep_util_13,glu_prod_13,bg_14,ins_abs_13,basal_bal_13,ins_indep_13,ins_dep_13,endo_bal_13,cho_14,tot_bal_13,met_irr_13,meal_14,gut_abs_14,cho_bal_14,renal_cl_14,ins_indep_util_14,ins_dep_util_14,glu_prod_14,bg_15,ins_abs_14,basal_bal_14,ins_indep_14,ins_dep_14,endo_bal_14,cho_15,tot_bal_14,met_irr_14,meal_15,gut_abs_15,cho_bal_15,renal_cl_15,ins_indep_util_15,ins_dep_util_15,glu_prod_15,bg_16,ins_abs_15,basal_bal_15,ins_indep_15,ins_dep_15,endo_bal_15,cho_16,tot_bal_15,met_irr_15,meal_16,gut_abs_16,cho_bal_16,renal_cl_16,ins_indep_util_16,ins_dep_util_16,glu_prod_16,bg_17,ins_abs_16,basal_bal_16,ins_indep_16,ins_dep_16,endo_bal_16,cho_17,tot_bal_16,met_irr_16,meal_17,gut_abs_17,cho_bal_17,renal_cl_17,ins_indep_util_17,ins_dep_util_17,glu_prod_17,bg_18,ins_abs_17,basal_bal_17,ins_indep_17,ins_dep_17,endo_bal_17,cho_18,tot_bal_17,met_irr_17,meal_18,gut_abs_18,cho_bal_18,renal_cl_18,ins_indep_util_18,ins_dep_util_18,glu_prod_18,bg_19,ins_abs_18,basal_bal_18,ins_indep_18,ins_dep_18,endo_bal_18,cho_19,tot_bal_18,met_irr_18,meal_19,gut_abs_19,cho_bal_19,renal_cl_19,ins_indep_util_19,ins_dep_util_19,glu_prod_19,bg_20,ins_abs_19,basal_bal_19,ins_indep_19,ins_dep_19,endo_bal_19,cho_20,tot_bal_19,met_irr_19,meal_20,gut_abs_20,cho_bal_20,renal_cl_20,ins_indep_util_20,ins_dep_util_20,glu_prod_20,bg_21,ins_abs_20,basal_bal_20,ins_indep_20,ins_dep_20,endo_bal_20,cho_21,tot_bal_20,met_irr_20,meal_21,gut_abs_21,cho_bal_21,renal_cl_21,ins_indep_util_21,ins_dep_util_21,glu_prod_21,bg_22,ins_abs_21,basal_bal_21,ins_indep_21,ins_dep_21,endo_bal_21,cho_22,tot_bal_21,met_irr_21,meal_22,gut_abs_22,cho_bal_22,renal_cl_22,ins_indep_util_22,ins_dep_util_22,glu_prod_22,bg_23,ins_abs_22,basal_bal_22,ins_indep_22,ins_dep_22,endo_bal_22,cho_23,tot_bal_22,met_irr_22,meal_23,gut_abs_23,cho_bal_23,renal_cl_23,ins_indep_util_23,ins_dep_util_23,glu_prod_23,bg_24,ins_abs_23,basal_bal_23,ins_indep_23,ins_dep_23,endo_bal_23,cho_24,tot_bal_23,met_irr_23,meal_24"

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
