import pandas as pd
input_datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
                "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
for input_dataset in input_datasets:

    file1 = pd.read_csv(f'./Original_Prompt_Causality/question/questions_{input_dataset}.csv')
    file2 = pd.read_csv(f'./Prompt/{input_dataset}_input.csv')


    selected_columns_file1 = file1.iloc[:, [0, 1]]
    selected_column_file2 = file2.iloc[:, 0]

    combined = selected_columns_file1.astype(str).agg(','.join, axis=1) + ' ' + selected_column_file2.astype(str)

    combined.to_csv(f'./Prompt_Final/Combined_{input_dataset}.csv', index=False, header=["prompt"])
    print(combined)


