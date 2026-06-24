from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# 加载GSM8K数据集


# 加载Llama模型和Tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # 替换为LLAMA-7B模型路径
model = AutoModelForCausalLM.from_pretrained(model_name,force_download=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 将模型移动到GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
dataset = load_dataset("gsm8k", split="test[:10]")  # 只取前10个样本进行测试
# 确保输入可以被模型处理
tokenizer.pad_token = tokenizer.eos_token

# 定义函数来解码模型的输出
def solve_math_problem(question):
    # 将问题转换为输入格式
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # 将输入移动到GPU（如果可用）
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 使用模型生成答案
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # 解码模型输出
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# 测试部分样本
for example in dataset:
    question = example["question"]
    print(f"问题：{question}")
    answer = solve_math_problem(question)
    print(f"回答：{answer}")
    print("=" * 50)
