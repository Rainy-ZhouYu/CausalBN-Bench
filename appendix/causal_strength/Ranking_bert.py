import os
import pandas as pd
import requests

# models_from_coauthors = [
#         'bert_base_mnli_ft',
#         'bert_base_mnli',
#         'bert_base',
#         'bert_large_mnli',
#         'bert_large',
#         'deberta_large_mnli',
#         'deberta_xlarge_mnli',
#         'distilbart_mnli',
#         'distilbert_mnli_42.06',
#         'distilbert_mnli',
#         'huggingface_mnli',
#         'longformer_base',
#         'random_majority',
#         'random_proportional',
#         'random_uniform',
#         'roberta_base_mnli',
#         'roberta_base',
#         'roberta_large_mnli', # best
#         'roberta_large',
#         'roberta_mnli',
#         'llama030',
#         'llama013',
#         'llama065',
#         'llama007',
#         'alpaca007',
#     ]

# 读取CSV文件
api_endpoint = "https://api-inference.huggingface.co/models/bert-base-uncased"
api_key = os.environ.get("HUGGINGFACE_API_KEY")

df = pd.read_csv("generate_question/question/questions_asia_v2.csv")
questions = df['prompt'].tolist()

answers = []
i = 0

for question in questions:
    question = question.rstrip('\\n')
    payload = {"question": question,
        "context": "Your context for the question"
    }
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(api_endpoint, json=payload, headers=headers)
    answer = response.json()
    answers.append(answer)

    i = i + 1
    print(i)
    print(answer)


qa_pairs = pd.DataFrame({'Question': questions, 'Answer': answers})
qa_pairs.to_csv("Response/asia_bert.csv", index=False)
