import os
import openai
import pandas as pd
import csv


def answer_for_gpt(questions, model, outputFileName):
    i = 0
    answers = []
    for question in questions:
        response = openai.ChatCompletion.create(
            model=f"{model}",  # 使用适当的模型名称
            messages=[
                {"role": "system", "content": "You are a highly intelligent question-answering bot with profound knowledge of causal inference."},
                {"role": "user", "content": question}
            ],
            max_tokens=100
        )
        answer = response.choices[0].message['content'].strip()
        answers.append(answer)
        i = i+1
        print(i)
        print(question)
        print(answer)
        data = [i, question, answer]
        # with open(outputFileName, 'a') as file:
        #     file.write(answer + '\n')
        with open(outputFileName, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入标题
            #writer.writerow(['order', 'Question', 'Answer'])
            writer.writerow([i, question, answer])

    return answers


openai.api_base = "https://api.gptapi.us/v1/"
openai.api_key = os.environ.get("OPENAI_API_KEY")


########  Gpt3.5 and Gpt4-based model######################
models = ['gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo']
# models = ['gpt-4-1106-preview']
# models = ['gpt-3.5-turbo']
# models = ['gpt-4']
input_datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
               "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
# input_datasets = ["water", "hailfinder", "hepar2", "win95pts"]
# input_datasets = ["mildew", "hepar2", "win95pts"]
# input_datasets = ["barley", "child", "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
# 对每个问题使用GPT模型生成回答


for model in models:
    Answer = []
    for input_dataset in input_datasets:
        outputFileName = f'Response/Rank_Process/{model}_{input_dataset}.csv'
        questions_df = pd.read_csv(f'Ranking/ranking_{input_dataset}.csv')  # 假设问题在第一列
        questions = questions_df['prompt'].tolist()
        # questions = questions[3572:] for gpt-3.5 turbo in mildew
        questions = questions
        Answer = answer_for_gpt(questions, model, outputFileName)


        qa_pairs = pd.DataFrame({'Question': questions, 'Answer': Answer})
        qa_pairs.to_csv(f'Response/Rank/{model}_{input_dataset}.csv', index=False)