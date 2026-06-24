import pandas as pd
import os

def create_combinations(csv_file_path, output_file_path):
    try:
        # Load the CSV file
        data = pd.read_csv(csv_file_path)

        # Extract the fourth column
        column_data = data.iloc[:, 3]

        # Initialize an empty list to store the combinations
        combinations = []

        # Generate combinations: each row with every other row
        for i in range(len(column_data)):
            for j in range(len(column_data)):
                if i != j:
                    combinations.append("The knowledge is given as follows. " + column_data.iloc[i] + " Meanwhile, " + column_data.iloc[j])

        # Convert the combinations to a DataFrame
        combinations_df = pd.DataFrame(combinations, columns=['Combinations'])

        # Ensure the directory of the output file exists, if not, create it
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Save the combinations to the output CSV file
        combinations_df.to_csv(output_file_path, index=False)

        return "Combinations created and saved successfully."

    except Exception as e:
        return str(e)

# Example usage
input_datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
                "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
for input_dataset in input_datasets:
    csv_file_path = f'Knowledge_Process/gpt-4-1106-preview_Knowledge_{input_dataset}.csv'  # Replace with the path to your CSV file
    output_file_path = f'Prompt/{input_dataset}.csv'  # Replace with the desired output file path

    result = create_combinations(csv_file_path, output_file_path)
    print(result)
