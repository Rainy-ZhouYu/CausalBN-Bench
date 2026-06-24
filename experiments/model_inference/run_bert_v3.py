## python file that generates predictions from a pretrained language model from huggingface transformers
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import csv
import pandas as pd
import torch
from transformers import RobertaForCausalLM, AutoTokenizer




def main(locationLlamaHF,outputFileName,inputFileName):
    tokenizer = AutoTokenizer.from_pretrained(locationLlamaHF, cache_dir="~/cache/")
    model = RobertaForCausalLM.from_pretrained(locationLlamaHF, offload_folder="offload", torch_dtype=torch.float16, cache_dir="~/cache/")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
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
            prompts=prompts[0]+"\nAnswer:"
            inputs = tokenizer([prompts], return_tensors='pt').to(device)

            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample= False,
                max_new_tokens=10,
                temperature=0,
            )
            outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            outputs=[[el] for el in outputs]
            with open(outputFileName, 'a') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                writer.writerows(outputs)

if __name__ == '__main__':
    ## model location HuggingFace format
    locationLlamaHF = "./LLM_Model/roberta_base"
    outputFileName = "Result/roberta-base.csv"
    inputFileName = "generate_question/question/questions_asia.csv"
    main(locationLlamaHF, outputFileName, inputFileName)