import openai
import pandas as pd
import csv


def answer_for_gpt(questions, model, outputFileName):
    i = 0
    answers = []
    for question in questions:
        response = openai.ChatCompletion.create(
            model=f"{model}",
            messages=[
                {"role": "system", "content": "You are a highly intelligent question-answering bot with profound knowledge of causal inference."},
                {"role": "user", "content": question}
            ],
            max_tokens = 1000
        )
        answer = response.choices[0].message['content'].strip()
        answers.append(answer)
        i = i+1
        print(i)
        print(question)
        # print(answer)
        data = [i, question, answer]
        # with open(outputFileName, 'a') as file:
        #     file.write(answer + '\n')
        with open(outputFileName, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            #writer.writerow(['order', 'Question', 'Answer'])
            writer.writerow([i, question, answer])

    return answers


openai.api_base = "openai_url"
openai.api_key = 'your_key'



models = ['gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo']

input_datasets = ["barley", "child",
               "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]



for model in models:
    Answer = []
    for input_dataset in input_datasets:
        outputFileName = f'{model}_{input_dataset}.csv'
        questions_df = pd.read_csv(f'ranking_{input_dataset}.csv')
        questions = questions_df['prompt'].tolist()

        questions = questions
        Answer = answer_for_gpt(questions, model, outputFileName)


        qa_pairs = pd.DataFrame({'Question': questions, 'Answer': Answer})
        qa_pairs.to_csv(f'Rank/{model}_{input_dataset}.csv', index=False)
