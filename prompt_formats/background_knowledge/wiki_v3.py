import csv
import requests


def query_wikipedia(word):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": word,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
    }
    response = requests.get(url, params=params)
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    extract = page.get("extract", "No description available.")

    return extract


def process_csv(input_file_path, output_file_path):
    with open(input_file_path, mode='r', encoding='utf-8') as infile, \
            open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            word = row[0]
            description = query_wikipedia(word)
            row.append(description)
            writer.writerow(row)


# 示例用法
dataset = 'alarm'
input_file_path = f'{dataset}.csv'
output_file_path = f'{dataset}_description.csv'
process_csv(input_file_path, output_file_path)


