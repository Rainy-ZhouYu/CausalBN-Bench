import os
import openai
import pandas as pd
import time
import requests

response = requests.get('https://api.openai.com/', timeout=20)  # 设置超时为20秒
##
# gpt_model_name2engine_name = {
#     'gpt_a': 'ada',
#     'gpt_b': 'babbage',
#     'gpt_c': 'curie',
#     'gpt_d': 'davinci',
#     'gpt_d001': 'text-davinci-001',
#     'gpt_d002': 'text-davinci-002',
#     'gpt_d003': 'text-davinci-003',
#     'gpt_d003cot': 'text-davinci-003',
#
#     'gpt3.5': "gpt-3.5-turbo",
#     'gpt4': "gpt-4",
#
#     'gpt_a_cls_10k_ft': 'ada:ft-academicszhijing:causalnli-cls-10k-2022-10-29-12-08-18',
#     'gpt_b_cls_10k_ft': 'babbage:ft-academicszhijing:causalnli-cls-10k-2022-10-29-13-10-17',
#     'gpt_c_cls_10k_ft': 'curie:ft-academicszhijing:causalnli-cls-10k-2022-10-29-12-34-57',
#     'gpt_d_cls_10k_ft': 'davinci:ft-academicszhijing:causalnli-cls-10k-2022-11-01-12-44-59',
# }
# 设置OpenAI API密钥
#  openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")

# 读取包含问题的CSV文件

# 用于存储回答的列表
answers = []
gpt_model_name2engine_name = {
    'gpt_a': 'ada',
    'gpt_b': 'babbage',
    'gpt_c': 'curie',
    'gpt_d': 'davinci',
    'gpt_d001': 'text-davinci-001',
    'gpt_d002': 'text-davinci-002',
    'gpt_d003': 'text-davinci-003',
    'gpt_d003cot': 'text-davinci-003'}
# 对于CSV文件中的每个问题，使用GPT模型生成回答


openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")

# 读取包含问题的CSV文件
questions_df = pd.read_csv('generate_question/question/questions_asia_v3.csv')  # 假设问题在第一列
questions = questions_df['prompt'].tolist()  # 'prompt'是列标题
#questions = questions_df[0].tolist()

# 用于存储回答的列表
openai.api_request_timeout = 20
# 对每个问题使用GPT模型生成回答
i = 0
for question in questions:
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # 使用适当的模型名称
            prompt=f"Q: {question}\nA:",
            max_tokens=30
        )
        answer = response.choices[0].text.strip()
        answers.append(answer)
        i = i+1
        print(i)
        print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")
        answers.append("Error generating response")



# 将问题和回答保存到新的CSV文件
qa_pairs = pd.DataFrame({'Question': questions, 'Answer': answers})
qa_pairs.to_csv('Response/asia_gpt_d003.csv', index=False)