## python file that generates predictions from a pretrained language model from huggingface transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import pandas as pd

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



def main(locationLlamaHF,outputFileName,inputFileName):
    tokenizer = AutoTokenizer.from_pretrained(locationLlamaHF, cache_dir="~/cache/",trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(locationLlamaHF, device_map="auto", offload_folder="offload", torch_dtype=torch.float16, cache_dir="~/cache/", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## read prompts from csv and generate predictions
    df=pd.read_csv(inputFileName)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    with open(outputFileName, 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        row=['pred']
        writer.writerow(row)

    with torch.no_grad():
        for i in range(0,df.shape[0],1):
            prompts=list(df['prompt'].values)[i:i+1]
            if model == "LLAMA7B" or model =="LLAMA13B" or model == "LLAMA30B":
                prompts=prompts[0]+"\nAnswer:"
            else:
                prompts = generate_prompt(prompts[0])
            inputs = tokenizer([prompts], return_tensors='pt').to(device)

            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False,
                max_new_tokens=10,
                temperature=1
            )
            outputs=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            outputs=[[el] for el in outputs]
            with open(outputFileName, 'a') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                writer.writerows(outputs)


if __name__ == '__main__':
    ## model location HuggingFace format
    # locationLlamaHF = "./LLAMA7B"
    # outputFileName = "Result/llama_asia.csv"
    # inputFileName = "generate_question/question/questions_asia_v3.csv"
    # main(locationLlamaHF,outputFileName,inputFileName)
    runtimes = [1,2,3,4,5]
    for runtime in runtimes:

        models = ["LLAMA7B", "LLAMA13B", "LLAMA30B", "OPT6d7b", "OPT2d7b", "OPT1d3b", "Internlm7b", "Internlm20b"]
        input_files = [f"./question_CoT.csv"]


        for model in models:
            for input_file in input_files:
                if model == "OPT6d7b" or model =="OPT2d7b" or model =="OPT1d3b":
                    locationLlamaHF = f"/path/to/local/models/{model}"
                    outputFileName = f"./Result/Task_CoT/{model}_{runtime}_{input_file.split('/')[-1].split('.')[0]}.csv"
                else:
                    locationLlamaHF = f"./{model}"
                    outputFileName = f"./Result/Task_CoT/{model}_{runtime}_{input_file.split('/')[-1].split('.')[0]}.csv"
                main(locationLlamaHF, outputFileName, input_file)
