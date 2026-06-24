import pandas as pd

# Assuming the CSV file is named 'example.csv' and is located in the same directory as this script
file_path = 'Knowledge_Process/gpt-4-1106-preview_Knowledge_alarm.csv'

# Load the CSV file
try:
    data = pd.read_csv(file_path)

    # Extract the fourth column
    column_data = data.iloc[:, 3]

    # Combine the first row with each of the other rows
    combined_data = [column_data.iloc[0] + " " + row for row in column_data[1:]]

    print(combined_data)
except Exception as e:
    error_message = str(e)
    combined_data = None
    error_message