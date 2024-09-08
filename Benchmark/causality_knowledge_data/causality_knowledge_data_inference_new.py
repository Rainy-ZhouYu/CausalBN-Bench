import numpy as np
import pandas as pd
import openai
import csv
import time
import json
openai.api_base = "openai_url"
openai.api_key = 'your_openai_key'
def split_matrix(matrix, n_splits):
    return np.array_split(matrix, n_splits, axis=0)

def create_chat_message(matrix_part, column_names, input_dataset, num):
    header = ' | '.join(column_names[:matrix_part.shape[1]])
    matrix_description = header + '\n'
    matrix_description += '\n'.join([' | '.join(map(str, row)) for row in matrix_part])
    return {"role": "user", "content": f"There are a total of five parts matrices, together forming an input sample of the Bayesian network named {input_dataset}. \
    You need to combine the five input matrix into a unified input sample of the Bayesian network matrix in the order of input.\
     Here is the part {num+1} of the matrix:\n{matrix_description}"}

def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            row = [float(num) for num in line.split()]
            matrix.append(row)
        return np.array(matrix)



def main():

    models = ['gpt-4-1106-preview']
    input_datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
                      "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
    for model in models:

        for input_dataset in input_datasets:
            df = pd.read_csv(f'questions_{input_dataset}.csv')
            questions = df['prompt'].tolist()
            outputFileName = f'{model}_{input_dataset}.csv'
            questions = questions
            Is_load_json = True

            if Is_load_json == False:
                with np.load(f'npz_file/{input_dataset}.npz', allow_pickle=True) as data:
                    matrix = data['matrix_num']
                    column_names = data['node_order']

                    print(matrix)
                    print(column_names)
                data.close()


                parts = split_matrix(matrix, int(matrix.__len__()/100))
                messages = []

                num = 0
                chat_log = None
                for part in parts:
                    message = create_chat_message(part, column_names, input_dataset, num)
                    print('message:', message)
                    messages.append(message)
                    response = openai.ChatCompletion.create(
                        model=f"{model}",  # 确保使用正确的模型名称
                        messages=messages,
                        api_key=openai.api_key
                    )
                    chat_log = response['choices'][0]['message']['content']
                    messages.append({"role": "assistant", "content": chat_log})
                    print('chat_log:', chat_log)
                    num = num + 1




                with open(f'data_description/{input_dataset}.json', 'w') as f:
                    json.dump(messages, f)

            else:
                with open(f'data_description/{input_dataset}.json', 'r') as f:
                    messages = json.load(f)

            i = 0
            answers = []
            for question in questions:
                input_prompt = messages.copy()
                input_prompt.append({"role": "user", "content": question})
                response = openai.ChatCompletion.create(
                    model=f"{model}",
                    messages=input_prompt,
                    api_key=openai.api_key
                )
                answer = response.choices[0].message['content'].strip()
                answers.append(answer)
                i = i + 1
                print(i)
                print(question)
                input_prompt.clear()
                data = [i, question, answer]
                with open(outputFileName, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([i, question, answer])
                if i % 5 == 0:
                    time.sleep(5)


if __name__ == "__main__":
    main()
