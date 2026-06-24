import os
import pandas as pd
import requests
import json

# 读取CSV文件

df = pd.read_csv("generate_question/question/questions_asia.csv")
questions = df['prompt'].tolist()
# Hugging Face API 设置
api_url = "https://api-inference.huggingface.co/models/bert-base-uncased"
api_key = os.environ.get("HUGGINGFACE_API_KEY")  # 用你的模型路径替换
headers = {"Authorization": f"Bearer {api_key}"}  # 用你的API密钥替换

# 发送请求并获取响应
for question in questions:
    response = requests.post(
        api_url,
        headers=headers,
        data=json.dumps({"inputs": question})  # 确保输入是正确格式的JSON字符串
    )
    # 处理API的响应
    print(response.json())


# api_endpoint = "https://api-inference.huggingface.co/models/bert-base-uncased"
# api_key = os.environ.get("HUGGINGFACE_API_KEY")

# 读取CSV文件
 # 替换为你的列名

# 将问题和回答添加到DataFrame，并保存到新的CSV文件
qa_pairs = pd.DataFrame({'Question': questions, 'Answer': answers})
qa_pairs.to_csv("Response/asia_bert.csv", index=False)
