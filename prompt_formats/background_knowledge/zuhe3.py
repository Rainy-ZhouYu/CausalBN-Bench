import pandas as pd
import os
def multiply_csv_rows(csv_file_path, output_file_path, num_copies):
    try:
        # Load the CSV file
        data = pd.read_csv(csv_file_path)

        # Create a list with multiple copies of the data
        duplicated_data = pd.concat([data] * num_copies)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        # Save the duplicated data to a new CSV file
        duplicated_data.to_csv(output_file_path, index=False)

        return "CSV rows duplicated and saved successfully."

    except Exception as e:
        return str(e)

input_datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
                "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
for input_dataset in input_datasets:
    csv_file_path = f'./Prompt/{input_dataset}.csv'  # Replace with the path to your CSV file
    output_file_path = f'./Prompt/{input_dataset}_input.csv'  # Replace with the desired output file path

    result = multiply_csv_rows(csv_file_path, output_file_path, 5)  # Multiply the rows 6 times
    print(result)
