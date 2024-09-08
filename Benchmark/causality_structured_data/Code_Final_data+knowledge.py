import numpy as np
import pandas as pd
import openai
import csv
import time
import json
openai.api_base = "https://api.gptapi.us/v1/"
openai.api_key = 'sk-6qA6GtGVoV1rbfO6B2714c847d4a4aD683113568110eEd62'

def split_matrix(matrix, n_splits):
    """将矩阵分割成 n 个部分"""
    return np.array_split(matrix, n_splits, axis=0)  # 按列分割

def create_chat_message(matrix_part, column_names, input_dataset, num):
    """为矩阵部分创建聊天消息，包括每列的变量名"""
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
            # 假设每行的数字是以两个空格分隔的
            row = [float(num) for num in line.split()]
            matrix.append(row)
        return np.array(matrix)



def main():
    # 示例矩阵
    # file_path = 'Dataset/0_alarm_data/Alarm1_s500_v1.txt'
    # matrix = read_matrix_from_file(file_path)
    # print(matrix)
    # column_names = ["HYPOVOLEMIA", "Left Ventricular End-Diastolic Volume", "STROKEVOLUME", "Central Venous Pressure", "Pulmonary Capillary Wedge Pressure", "Left Ventricular Failure", "HISTORY", "Cardiac Output", "ERRLOWOUTPUT", "HRBP", "ERRCAUTER", "Heart Rate EKG", "Heart Rate Saturation", "Insufficient Anesthesia", "CATECHOL", "ANAPHYLAXIS", "Total Peripheral Resistance", "Blood Pressure", "KINKEDTUBE", "PRESS", "VENTLUNG", "Fraction of Inspired Oxygen", "Peripheral Venous Saturation", "Arterial Oxygen Saturation", "Pulmonary Embolus", "Pulmonary Artery Pressure", "SHUNT", "INTUBATION", "Minimum Volume", "Ventilation Alveolar", "DISCONNECT", "Ventilation Tube", "Minimum Volume Set", "Ventilation Machine", "Expired CO2", "Arterial CO2","Heart Rate"]

    models = ['gpt-4-1106-preview']
    #models = ['gpt-3.5-turbo-1106']
    input_datasets = ["insurance", "mildew", "water","alarm", "barley", ]
    # input_datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
    #                   "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
    # input_datasets = ["earthquake", "sachs", "survey", "alarm", "barley", "child",
    #                   "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
    # 读取问题
    for model in models:

        for input_dataset in input_datasets:
            df = pd.read_csv(f'question_knowledge/Prompt_Final/Combined_{input_dataset}.csv')
            questions = df['prompt'].tolist()
            rows_per_class = len(questions) // 5
            outputFileName = f'D:/BaiduNetdiskWorkspace/周黑娃的文件夹/Causality/Causal+LLM/corr2cause-main/code/Response/Task_Data_Knowledge/Task_Data_Knowledge_Causality_Process/{model}_{input_dataset}.csv'
            if input_dataset == 'alarm':
                questions = questions[rows_per_class*2+157:rows_per_class*3]
                Is_load_json = False
            elif input_dataset == 'insurance':
                questions = questions[rows_per_class*2+66:rows_per_class*3]
                Is_load_json = False
            else:
                questions = questions[rows_per_class*2:rows_per_class*3]
                Is_load_json = False

            if Is_load_json == False:
                with np.load(f'npz_file/{input_dataset}.npz', allow_pickle=True) as data:
                    matrix = data['matrix_num']
                    column_names = data['node_order']

                    print(matrix)
                    print(column_names)
                data.close()

                # 分割矩阵
                parts = split_matrix(matrix, int(matrix.__len__()/100))
                messages = []
                # 初始化聊天
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

                # 发送问题并获取回答

                # 将列表保存到 JSON 文件
                with open(f'data_description_knowledge/{input_dataset}.json', 'w') as f:
                    json.dump(messages, f)
                # with open(f'{input}.json', 'w') as file:
                #     json.dump(messages, file)
            else:
                with open(f'data_description_knowledge/{input_dataset}.json', 'r') as f:
                    messages = json.load(f)

            i = 0
            answers = []
            for question in questions:
                input_prompt = messages.copy()
                input_prompt.append({"role": "user", "content": question})
                response = openai.ChatCompletion.create(
                    model=f"{model}",  # 使用正确的模型名称
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
                if i % 2 == 0:
                    time.sleep(4)


if __name__ == "__main__":
    main()
