import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# 读取CSV文件
df = pd.read_csv('your_file.csv')  # 将 'your_file.csv' 替换为你的文件名

# 确保 'prompt' 和 'label' 列存在
if 'prompt' in df.columns and 'label' in df.columns:
    # 取出预测值和真实标签
    predictions = df['prompt']
    labels = df['label']

    # 计算F1分数和准确率
    f1 = f1_score(labels, predictions, average='binary', pos_label='yes')
    accuracy = accuracy_score(labels, predictions)

    # 打印结果
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')
else:
    print("Error: CSV file must contain 'prompt' and 'label' columns")
