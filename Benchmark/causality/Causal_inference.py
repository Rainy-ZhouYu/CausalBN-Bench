import argparse
import openai
import pandas as pd
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


def answer_for_gpt(questions, model, outputFileName):
    i = 0
    answers = []
    for question in questions:
        attempts = 0
        success = False
        while attempts < 3 and not success:  
            try:
                response = openai.ChatCompletion.create(
                    model=f"{model}",
                    messages=[
                        {"role": "system",
                         "content": "You are a highly intelligent question-answering bot with profound knowledge of causal inference."},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=100
                )
                answer = response.choices[0].message['content'].strip()
                answers.append(answer)
                i = i + 1
                print(i)
                print(question)
                print(answer)
                data = [i, question, answer]
                with open(outputFileName, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([i, question, answer])
                success = True
            except openai.error.OpenAIError as e:
                print(f"An error occurred: {e}")
                attempts += 1
                time.sleep(5)  

        if not success:
            print(f"Failed to get an answer for the question: {question}")

    return answers



def hf_main(locationLlamaHF, outputFileName, inputFileName):
    tokenizer = AutoTokenizer.from_pretrained(locationLlamaHF, cache_dir="~/cache/")
    model = AutoModelForCausalLM.from_pretrained(locationLlamaHF, device_map="auto", offload_folder="offload",
                                                 torch_dtype=torch.float16, cache_dir="~/cache/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(inputFileName)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    with open(outputFileName, 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        writer.writerow(['pred'])

    with torch.no_grad():
        for i in range(0, df.shape[0], 1):
            prompts = list(df['prompt'].values)[i:i + 1]
            prompts = generate_prompt(prompts[0])
            inputs = tokenizer([prompts], return_tensors='pt').to(device)

            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False,
                max_new_tokens=10,
                temperature=1
            )
            outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            outputs = [[el] for el in outputs]
            with open(outputFileName, 'a') as csvoutput:
                writer.writerows(outputs)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a model type: 'GPT' or 'Huggingface'")
    parser.add_argument('--model_type', type=str, required=True, help="Choose between 'GPT' or 'Huggingface'")
    args = parser.parse_args()

    if args.model_type == 'GPT':
        openai.api_base = "opanai_url"
        openai.api_key = 'your_key'
        models = ['gpt-3.5-turbo-1106']
        input_datasets = ["water", "hailfinder"]

        for model in models:
            Answer = []
            for input_dataset in input_datasets:
                outputFileName = f'/{model}_{input_dataset}.csv'
                questions_df = pd.read_csv(f'questions_{input_dataset}.csv')
                questions = questions_df['prompt'].tolist()
                Answer = answer_for_gpt(questions, model, outputFileName)

                qa_pairs = pd.DataFrame({'Question': questions, 'Answer': Answer})
                qa_pairs.to_csv(f'Response/Task_Causal/{model}_{input_dataset}.csv', index=False)

    elif args.model_type == 'Huggingface':
        models = ["OPT1d3b", "OPT2d7b", "OPT6d7b"]#open LLMs
        input_files = [f"questions_{region}.csv" for region in ["asia"]]

        for model in models:
            for input_file in input_files:
                locationLlamaHF = f"{model}"
                outputFileName = f"{model}_{input_file.split('/')[-1].split('.')[0]}.csv"
                hf_main(locationLlamaHF, outputFileName, input_file)
    else:
        print("Invalid model type! Please choose either 'GPT' or 'Huggingface'.")
