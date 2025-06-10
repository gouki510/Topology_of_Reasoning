from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Dict, Optional, Sequence
import transformers
import torch
from torch.utils.data import DataLoader
import os, random
import numpy as np
from huggingface_hub import login
from vllm import LLM, SamplingParams

from load_data.preprocess import GSMData, MathData, AquaData, SVAMPData, MATH_500Data, AIME_DATA
from load_data.k_shot_dataset import KshotDataset
import calculator
from model.generation_utils import make_sparse_mask
from model.utils import model_name_mapping

INVALID_ANS = "[invalid]"


GSMK_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.
The last line of your response should be of the following format: 'The answer is: ANSWER.' (without quotes) where ANSWER is just the final number or expression that solves the problem.

{Question}
""".strip()


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="gpt2")
    base_model_name_or_path: Optional[str] = field(default="gpt2")
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None, metadata={"help": "Path to the output dir."})
    max_length: Optional[int] = field(default=512)
    decoding_scheme: Optional[str] = field(default="greedy")
    load_in_8bit: Optional[bool] = field(default=False)
    use_calculator: Optional[bool] = field(default=False)
    parameter_efficient_mode: Optional['str'] = field(default='none', metadata={"choices": ["none", "prompt-tuning", "lora", "lora+prompt-tuning"]})
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Require for llama family."})
    enable_cpu_offload: Optional[bool] = field(default=False)
    flash_attention: Optional[bool] = field(default=True)

@dataclass
class DataArguments:
    dataset: str = field(default=None, metadata={"help": "dataset name."})
    batch_size: Optional[int] = field(default=16)
    use_demonstrations: Optional[bool] = field(default=False)
    demo_selection: Optional[str] = field(default="uniform")
    candidate_size: Optional[int] = field(default=100)
    k_shot: Optional[int] = field(default=4)
    seed: Optional[int] = field(default=42)
    num_test: Optional[int] = field(default=1000)
    prompt_template: Optional[str] = field(default=None)
    embedding_model_name: Optional[str] = field(default='all-mpnet-base-v2')


def main():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    random.seed(data_args.seed)

    if model_args.output_dir is None:
        model_args.output_dir = os.path.join('results', model_args.model_name_or_path)
    
    os.makedirs(model_args.output_dir, exist_ok = True)
    print(model_args.model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    print("loaded tokenizer")

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if model_args.parameter_efficient_mode != 'none':
        model_name = model_name_mapping(model_args.base_model_name_or_path)
    else:
        model_name = model_args.model_name_or_path

    if model_args.load_in_8bit:
        quantization_config = transformers.BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=model_args.enable_cpu_offload)

        model = LLM(model=model_args.model_name_or_path,
                    max_model_len=model_args.max_length,
                    dtype=torch.float16,
                    )
        
    print("loaded model.")


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.eval()  
    
    if data_args.dataset == "gsm8k":
        data_class = GSMData
    elif data_args.dataset == "math":
        data_class = MathData
    elif data_args.dataset == "aqua":
        data_class = AquaData
    elif data_args.dataset == "svamp":
        data_class = SVAMPData
    elif data_args.dataset == "math_500":
        data_class = MATH_500Data
    elif data_args.dataset == "aime":
        data_class = AIME_DATA
    else:
        raise NotImplementedError

    dataset = data_class("test", [], 
                        prompt_template=data_args.prompt_template,
                        tokenizer=tokenizer,)

    if len(dataset) > data_args.num_test:
        idx = random.choices(list(range(len(dataset))), k=data_args.num_test)
        new_x = []
        new_y = []
        for i in idx:
            new_x.append(dataset[i]['x'])
            new_y.append(dataset[i]['y'])
        dataset.x = new_x
        dataset.y = new_y

    assert len(dataset) <= data_args.num_test
    print(dataset[0], len(dataset))

    if data_args.use_demonstrations:

        demo_dataset = data_class("train", [], 
                        prompt_template=data_args.prompt_template,
                        tokenizer=tokenizer,)
        rand_ids = random.sample(range(len(demo_dataset)), data_args.candidate_size)
        demo_dataset = [demo_dataset[i] for i in rand_ids]
        save_dir = f'demos/{data_args.dataset}/gpt2-xl' #Llama-2-70b-hf

        if os.path.exists(save_dir + '/sorted_demo_data.json') or data_args.demo_selection != 'prompt':
            dataset = KshotDataset(dataset, demo_dataset, data_args.k_shot,
                                data_args.demo_selection, save_dir=save_dir)
        else:
            dataset = KshotDataset(dataset, demo_dataset, data_args.k_shot,
                                    data_args.demo_selection, model, tokenizer, 
                                    None, save_dir)
            print("selected demos: ", dataset[0]['x'])
            print("prompt losses calculated")
            exit(0)
        
    
    dataloader = DataLoader(dataset, batch_size=data_args.batch_size, shuffle=False)

    if data_args.use_demonstrations:
        out_file_name = f'{model_args.output_dir}/{data_args.dataset}_test_cal={model_args.use_calculator}_demo={data_args.demo_selection}_k={data_args.k_shot}_output.txt'
    else:
        out_file_name = f'{model_args.output_dir}/{data_args.dataset}_test_cal={model_args.use_calculator}_output.txt'
            
    output = []
    num_correct = 0
    num_all = 0
    tmp_correct =0

    for i, batch in tqdm(enumerate(dataloader)):
        x_text, y_text = batch['x'], batch['y']
        if data_args.use_demonstrations:
            print("\n---x_text---")
            print(x_text)
            print("--------------------")
            encoding = tokenizer(x_text, padding=True, return_tensors='pt').to(device)
            max_length = min(model_args.max_length, encoding['input_ids'].size(1) + 512)
            sampling_params = SamplingParams(
                    max_tokens=max_length,
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.4,
                )
            with torch.no_grad():
                generated_outputs = model.generate(x_text, sampling_params=sampling_params)
                generated_ids = [generated_output.outputs[i].token_ids for i, generated_output in enumerate(generated_outputs)]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            print("---- generated_texts ----")
            print(generated_texts)
            print("--------------------")
                    
        else:
            x_text = [GSMK_QUERY_TEMPLATE.format(Question=x) for x in x_text]
            print("\n---x_text---")
            print(x_text)
            print("--------------------")
            print("\n---y_text---")
            print(y_text)
            print("--------------------")
            encoding = tokenizer(x_text, padding=True, return_tensors='pt').to(device)
            # max_length = min(model_args.max_length, encoding['input_ids'].size(1) + 512)
            max_length = model_args.max_length
            sampling_params = SamplingParams(
                max_tokens=max_length,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.4,
            )
            with torch.no_grad():
                generated_outputs = model.generate(x_text, sampling_params=sampling_params)
                generated_ids = [generated_output.outputs[i].token_ids for i, generated_output in enumerate(generated_outputs)]
            try:
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                print("---- generated_texts ----")
                print(generated_texts)
                print("--------------------")
            except:
                print("cannot decode: ")
                print(generated_ids)

        for text, x, y in zip(generated_texts, x_text, y_text):
            text, x, y = str(text), str(x), str(y)
            output.append((x, text, y, tmp_correct))
            num_all += 1

    # to csv
    import pandas as pd
    df = pd.DataFrame(output, columns=['Question', 'generated_text', 'Answer', 'correct'])
    df.to_csv(out_file_name.replace('.txt', '.csv'), index=False)


if __name__ == "__main__":
    main()