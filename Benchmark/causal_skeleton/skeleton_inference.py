import argparse
import openai
import pandas as pd
import csv
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def answer_for_gpt(questions, model, outputFileName):
    i = 0
    answers = []
    for question in questions:
        response = openai.ChatCompletion.create(
            model=f"{model}",
            messages=[
                {"role": "system", "content": "You are a highly intelligent question-answering bot with profound knowledge of causal inference and causal learning."},
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
        with open(outputFileName, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([i, question, answer])
    return answers

def hf_main(locationLlamaHF, outputFileName, inputFileName):
    tokenizer = LlamaTokenizer.from_pretrained(locationLlamaHF, cache_dir="~/cache/")
    model = LlamaForCausalLM.from_pretrained(locationLlamaHF, device_map="auto", offload_folder="offload", torch_dtype=torch.float16, cache_dir="~/cache/")
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
            prompts = prompts[0] + "\nAnswer:"
            inputs = tokenizer([prompts], return_tensors='pt').to(device)

            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False,
                max_new_tokens=100, temperature=0
            )
            outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            outputs = [[el] for el in outputs]
            with open(outputFileName, 'a') as csvoutput:
                writer.writerows(outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, help="Choose between 'GPT' or 'Huggingface'")
    args = parser.parse_args()

    if args.model_type == 'GPT':
        openai.api_base = "openai_url"
        openai.api_key = 'your_openai_key'
        models = ['gpt-4']
        input_datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child", "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
        for model in models:
            Answer = []
            for input_dataset in input_datasets:
                outputFileName = f'Task_Causal_Process/{model}_{input_dataset}.csv'
                questions_df = pd.read_csv(f'{input_dataset}.csv')
                questions = questions_df['prompt'].tolist()
                Answer = answer_for_gpt(questions, model, outputFileName)

                qa_pairs = pd.DataFrame({'Question': questions, 'Answer': Answer})
                qa_pairs.to_csv(f'Task_Causal/{model}_{input_dataset}.csv', index=False)

    elif args.model_type == 'Huggingface':
        models = ["LLAMA13b", "LLAMA7B", "LLAMA30B"]
        input_files = [f"Prompt/{region}.csv" for region in ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child", "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pt"]]
        for model in models:
            for input_file in input_files:
                locationLlamaHF = f"{model}"
                outputFileName = f"Task_Skeleton/{model}_{input_file.split('/')[-1].split('.')[0]}.csv"
                hf_main(locationLlamaHF, outputFileName, input_file)
    else:
        print("Invalid model type! Please choose either 'GPT' or 'Huggingface'.")
