import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import csv
import pandas as pd
import time
import requests


def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_ctxt}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def hf_main(locationLlamaHF, outputFileName, inputFileName):
    tokenizer = AutoTokenizer.from_pretrained(locationLlamaHF, cache_dir="~/cache/")
    model = AutoModelForCausalLM.from_pretrained(locationLlamaHF, device_map="auto", offload_folder="offload", cache_dir="~/cache/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(inputFileName)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    with open(outputFileName, 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        writer.writerow(['pred'])

    with torch.no_grad():
        for i in range(df.shape[0]):
            prompts = list(df['prompt'].values)[i:i+1]
            prompts = generate_prompt(prompts[0])
            inputs = tokenizer([prompts], return_tensors='pt').to(device)
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False,
                max_new_tokens=10, temperature=0
            )
            outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            outputs = [[el] for el in outputs]
            with open(outputFileName, 'a') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                writer.writerows(outputs)

# 第二个代码 GPT 模型的 main 函数
def gpt_main():
    openai.api_key = 'your_key'
    questions_df = pd.read_csv('generate_question/question/questions_asia_v3.csv')
    questions = questions_df['prompt'].tolist()
    answers = []
    openai.api_request_timeout = 20

    for i, question in enumerate(questions):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a highly intelligent question-answering bot with profound knowledge of causal learning and causal inference."},
                    {"role": "user", "content": question}
                ],
                max_tokens=100
            )
            answer = response.choices[0].message['content'].strip()
            answers.append(answer)
            print(i+1, answer)
        except Exception as e:
            print(f"An error occurred: {e}")
            answers.append("Error generating response")

    qa_pairs = pd.DataFrame({'Question': questions, 'Answer': answers})
    qa_pairs.to_csv('Response/asia_gpt_4.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, help='Choose between HuggingFace or GPT.')
    args = parser.parse_args()

    if args.model_type == 'Huggingface':
        models = ["Alpaca-7b"]
        input_files = ['/home/yzhou/CausalBenchmark/questions_alarm.csv']
        for model in models:
            for input_file in input_files:
                locationLlamaHF = f"/home/yzhou/CausalBenchmark/{model}"
                outputFileName = f"/home/yzhou/CausalBenchmark/Result/Task_Original/{model}_{input_file.split('/')[-1].split('.')[0]}.csv"
                hf_main(locationLlamaHF, outputFileName, input_file)
    elif args.model_type == 'GPT':
        gpt_main()
    else:
        print("Invalid model type! Please choose either 'Huggingface' or 'GPT'.")
