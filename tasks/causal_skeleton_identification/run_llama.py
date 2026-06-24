## python file that generates predictions from a pretrained language model from huggingface transformers
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import csv
import pandas as pd

def main(locationLlamaHF,outputFileName,inputFileName):
    tokenizer = LlamaTokenizer.from_pretrained(locationLlamaHF, cache_dir="~/cache/")
    model = LlamaForCausalLM.from_pretrained(locationLlamaHF, device_map="auto", offload_folder="offload", torch_dtype=torch.float16, cache_dir="~/cache/")
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
            prompts=list(df['Relationship'].values)[i:i+1]
            prompts=prompts[0]+"\nAnswer:"
            inputs = tokenizer([prompts], return_tensors='pt').to(device)

            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False,
                max_new_tokens=10,temperature=0
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

    models = ["LLAMA13B", "LLAMA7B", "LLAMA30B"]  # 假设的三个模型
    # input_files = [f"generate_question/question/questions_{region}.csv" for region in
    #                ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
    #                 "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pt"]]
    input_files = [f"./Skeleton/Prompt2/{region}.csv" for region in
                   ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
                    "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]]

    for model in models:
        for input_file in input_files:
            locationLlamaHF = f"./{model}"
            outputFileName = f"./Result/Task_Skeleton/{model}_{input_file.split('/')[-1].split('.')[0]}.csv"
            main(locationLlamaHF, outputFileName, input_file)
