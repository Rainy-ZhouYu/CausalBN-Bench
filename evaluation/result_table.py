import pandas as pd

# 定义数据
data = {
    'F1 Score': [0.8, 0.85, 0.9, 0.87, 0.88],  # 示例数据
    'Accuracy': [0.7, 0.75, 0.8, 0.77, 0.78],
    'SHD': [5, 4, 3, 3, 2],
    'SID': [10, 9, 8, 7, 6],
    'Edge Sparsity': [0.6, 0.65, 0.7, 0.68, 0.67]
}

# 创建 DataFrame
df = pd.DataFrame(data, index=['LLAMA', 'ALPHA', 'BERT', 'GPT3.5', 'GPT4'])

# 保存为 CSV 文件
df.to_csv('model_performance.csv')

print("CSV 文件已创建并保存。")
print(df)
