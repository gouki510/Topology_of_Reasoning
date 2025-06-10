import re, os, json, pickle
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
from load_data.utils import _strip_string, delete_extra_zero, compare_both_string_and_number_format
from transformers import T5EncoderModel, AutoTokenizer, AutoModelForCausalLM

class NodeType:
    def __init__(self, cluster_model, embedding_model_name, cache_dir=None):
        self.cluster_model = cluster_model
        self.tokenizer = None
        self.embedding_model_name = embedding_model_name
        if 't5' in embedding_model_name:
            self.embedding_model = T5EncoderModel.from_pretrained(
                embedding_model_name, cache_dir=cache_dir,).to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained(
                embedding_model_name, cache_dir=cache_dir, legacy=False)
        else:
            if 'phi-2' in embedding_model_name:
                embedding_model_name = "../llms/phi-2/"
            elif 'mistral' in embedding_model_name:
                embedding_model_name = "../llms/Mistral-7B-v0.1/"
            elif 'llama-2' in embedding_model_name:
                embedding_model_name = "../llms/llama2-hf/7b/"
            self.embedding_model = AutoModelForCausalLM.from_pretrained(
                embedding_model_name, trust_remote_code=True, cache_dir=cache_dir,
                torch_dtype=torch.float16, device_map="auto", load_in_8bit=True, 
                offload_folder="offload", offload_state_dict = True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                embedding_model_name, cache_dir=cache_dir, legacy=False)

        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = 0
        self.embedding_model.eval()

    def predict(self, text: str, start=0):
        with torch.no_grad():

            if self.embedding_model_name == 'all-mpnet-base-v2':
                embedding = self.embedding_model.encode(text.split('\n')[start:-1])

            elif 't5' in self.embedding_model_name:
                inputs = self.tokenizer(text.split('\n')[start:-1], return_tensors="pt", 
                    padding="longest", max_length=self.tokenizer.model_max_length, 
                    truncation=True,).to('cuda')
                outputs = self.embedding_model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                mean_hidden = torch.sum(last_hidden_states 
                    * inputs["attention_mask"].unsqueeze(-1), 1)\
                        /torch.sum(inputs["attention_mask"], -1).unsqueeze(-1)
                embedding = mean_hidden.cpu().numpy()

            else:
                inputs = self.tokenizer([text], return_tensors="pt", 
                    padding="longest", max_length=self.tokenizer.model_max_length, 
                    truncation=True,).to('cuda')
                # if 'token_type_ids' in inputs:
                #     del inputs['token_type_ids']
                outputs = self.embedding_model(**inputs, 
                    output_hidden_states=True, return_dict=True)
                last_hidden_state = outputs.hidden_states[-1][0]
                if 'llama' in self.embedding_model_name:
                    split_id = 13
                elif 'gpt2' in self.embedding_model_name or 'phi-2' in self.embedding_model_name:
                    split_id = 198
                else:
                    print("step split id not defined for ", self.embedding_model_name)
                    exit(1)
                step_mask = torch.cumsum(inputs['input_ids'][0]==split_id, dim=-1)
                print(step_mask)
                print(start)
                embedding = []
                for j in range(start, torch.max(step_mask)):
                    step_j_mask = (step_mask == j).int().float()
                    step_j_rep = (last_hidden_state * step_j_mask.unsqueeze(-1)).sum(0)
                    step_len = step_j_mask.sum()
                    if step_len > 0:
                        embedding.append((step_j_rep/step_len).cpu().numpy())
                    else:
                        print("current step is empty")
                embedding = np.stack(embedding, axis=0)
                print(embedding)
            # print(embedding)
                
            # if 'vae' in model_args.extract_step_type_tokens:
            embedding = torch.from_numpy(embedding).float().to('cuda')
            label = self.cluster_model.predict(embedding).cpu().numpy()
            # else:
                # label = self.cluster_model.predict(embedding)
            print(label)

        return label

class BaseGraph(Dataset):

    def __init__(self, dataset_name: str, split: str, cluster_model_name: str, embedding_model_name: str,
                num_nodes: int, data_dir='data', tokenizer=None, 
                invalid_ans="[invalid]", prompt_template=None, cache_dir=None):
        super(BaseGraph, self).__init__()
        
        self.dataset_name = dataset_name
        self.data_dir = f'{data_dir}/{self.dataset_name}/{split}/{embedding_model_name}'
        self.split = split
        self.cluster_model_name = cluster_model_name
        self.embedding_model_name = embedding_model_name
        self.num_nodes = num_nodes
        self.cache_dir = cache_dir
        self.INVALID_ANS = invalid_ans
        self.ANS_RE = re.compile(r"The answer is: (\-?[0-9\.\,]+)")
        self.tokenizer = tokenizer
        self.node_predictor = None

        if prompt_template is not None:
            file_name = f"./load_data/prompt_templates/{prompt_template}.json"
            assert os.path.exists(file_name), f"Prompt template {prompt_template} not found."
            self.prompt_template = json.load(open(file_name, "r"))
        else:
            self.prompt_template = None
        
        self.cots = json.load(open(f"{self.data_dir}/cot_text.json"))
        self.example_ids = np.load(f"{self.data_dir}/example_id.npy")
        print(self.example_ids)
        self.graph = []
        step_ids = np.arange(len(self.cots))
        self.node_ids = np.load(f"{self.data_dir}/{cluster_model_name}_{num_nodes}_clusters.npy")
        for i in range(num_nodes):
            self.graph.append(step_ids[self.node_ids == i])

        assert len(self.cots) == len(self.example_ids)

    def load_raw_data(self, split: str):
        raise NotImplementedError
    
    def parse_q_a(self, example):
        raise NotImplementedError      
    
    def __len__(self):
        return len(self.cots)

    def __getitem__(self, idx):
        return dict(cot=self.cots[idx], example_id=self.example_ids[idx], node_id=self.node_ids[idx])



class GSMGraph(BaseGraph):

    def __init__(self, split: str, cluster_model_name: str, embedding_model_name: str,
                num_nodes: int, data_dir='data', tokenizer=None, 
                invalid_ans="[invalid]", prompt_template=None, cache_dir=None):

        super(GSMGraph, self).__init__('gsm8k', split, cluster_model_name, embedding_model_name,
                num_nodes, data_dir, tokenizer, invalid_ans, prompt_template, cache_dir)

    def load_raw_data(self, split: str):
        return load_dataset('gsm8k', 'main')[split]
    
    def parse_q_a(self, example):
        cot, ans = example["answer"].split("####")
        cot = cot.strip()
        cot = cot.split('. ')
        cot_steps = []
        for step in cot:
            for s in step.strip().split('\n'):
                s = s.strip()
                if len(s) == 0:
                    continue
                if s[-1] != '.':
                    s += '.'
                cot_steps.append(s)
            
        ans = ans.strip()
        q = example["question"].strip()
        inst = 'Solve the following problem step by step, and give a numerical answer.'
        return inst, q, cot_steps, ans

    

class AquaGraph(BaseGraph):

    def __init__(self, split: str, cluster_model_name: str, embedding_model_name: str,
                num_nodes: int, data_dir='data', tokenizer=None, 
                invalid_ans="[invalid]", prompt_template=None, cache_dir=None):

        super(AquaGraph, self).__init__('aqua', split, cluster_model_name, embedding_model_name,
                num_nodes, data_dir, tokenizer, invalid_ans, prompt_template, cache_dir)

    def load_raw_data(self, split: str):
        return load_dataset('aqua_rat', 'raw')[split]
    
    def parse_q_a(self, example):
        choice = "(" + "(".join(e.strip() for e in example["options"])
        choice = choice.replace("(", " (").replace(")", ") ")
        choice = "Answer Choices:" + choice

        q = example["question"].strip() + '\n ' + choice

        ans = example["correct"].strip()
        cot = example["rationale"].strip()
        cot = cot.split('. ')
        cot_steps = []
        for step in cot:
            for s in step.strip().split('\n'):
                s = s.strip()
                if len(s) == 0:
                    continue
                cot_steps.append(s)

        if 'Explanation' in cot_steps[0]:
            print(cot_steps[0])
            cot_steps[0] = cot_steps[0][12:].strip()

        if len(cot_steps[0]) < 2:
            cot_steps = cot_steps[1:]
        
        if len(cot_steps) == 0:
            return None, None, None, None
        
        if 'Ans' in cot_steps[-1] or 'ANS' in cot_steps[-1] or 'ans' in cot_steps[-1] or 'Option' in cot_steps[-1]:
            print(cot_steps[-1])
            if len(cot_steps[-1]) < 30:
                cot_steps = cot_steps[:-1]
        elif len(cot_steps[-1]) < 5:
            print(cot_steps[-1])
            cot_steps = cot_steps[:-1]

        inst = 'Solve the following problem step by step, and choose the best answer from the given choices.'

        return inst, q, cot_steps, ans
        

class MathGraph(BaseGraph):

    def __init__(self, split: str, cluster_model_name: str, embedding_model_name: str,
                num_nodes: int, data_dir='data', tokenizer=None, problem_type=None,
                invalid_ans="[invalid]", prompt_template=None, cache_dir=None):

        super(MathGraph, self).__init__('math', split, cluster_model_name, embedding_model_name,
                num_nodes, data_dir, tokenizer, invalid_ans, prompt_template, cache_dir)
        
        self.type = problem_type

    def load_raw_data(self, split: str):
        all_data = load_dataset('competition_math')[split]
        filtered_data = []
        if self.type in ['Algebra', 'Geometry', "Counting & Probability", 
                    "Intermediate Algebra", "Number Theory", "Prealgebra",
                    "Precalculus"]:
            for ex in all_data:
                if ex['type'] == self.type:
                    filtered_data.append(ex)
            return filtered_data
        else:
            return all_data
    
    def parse_q_a(self, example):

        q = example["problem"].strip()
        cot = example["solution"].strip()
        cot = cot.split('. ')
        cot_steps = []
        for step in cot:
            for s in step.strip().split('\n'):
                s = s.strip()
                if len(s) == 0:
                    continue
                cot_steps.append(s)

        ans = self.extract_answer(example["solution"])

        inst = 'Solve the following problem step by step, and give the exact answer.'

        return inst, q, cot_steps, ans

    def extract_answer(self, completion):
        if'The answer is:' in completion:
            pred = completion.split('The answer is:')[-1].strip()
        elif'The answer is ' in completion:
            pred = completion.split('the answer is ')[-1].strip()
        elif'the answer is ' in completion:
            pred = completion.split('the answer is ')[-1].strip()
        elif 'boxed' in completion:
            ans = completion.split('boxed')[-1]
            if (ans[0] == '{'):
                stack = 1
                a = ''
                for c in ans[1:]:
                    if (c == '{'):
                        stack += 1
                        a += c
                    elif (c == '}'):
                        stack -= 1
                        if (stack == 0): break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            a = _strip_string(a)
            pred=a

        else:
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, completion)
            if(len(pred) >= 1):
                pred = pred[-1]
            else: pred = ''
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]
            if pred[-1] == "/":
                pred = pred[:-1]
        pred=_strip_string(pred)
        if 'boxed' in pred:
            ans = pred.split('boxed')[-1]
            if (ans[0] == '{'):
                stack = 1
                a = ''
                for c in ans[1:]:
                    if (c == '{'):
                        stack += 1
                        a += c
                    elif (c == '}'):
                        stack -= 1
                        if (stack == 0): break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            a = _strip_string(a)
            pred=a
        return pred
    

class SVAMPGraph(BaseGraph):
    
    def __init__(self, split: str, cluster_model_name: str, embedding_model_name: str,
                num_nodes: int, data_dir='data', tokenizer=None, 
                invalid_ans="[invalid]", prompt_template=None, cache_dir=None):

        super(SVAMPGraph, self).__init__('svamp', split, cluster_model_name, embedding_model_name,
                num_nodes, data_dir, tokenizer, invalid_ans, prompt_template, cache_dir)

    def load_raw_data(self, split: str):
        return load_dataset('ChilleD/SVAMP')[split]
    
    def parse_q_a(self, example):

        q = example["Body"].strip() + example["Question"].strip()
        cot_steps = [example["Equation"].strip()]

        ans = str(example["Answer"])

        inst = 'Solve the following problem by an equation, and give a numerical answer.'

        return inst, q, cot_steps, ans