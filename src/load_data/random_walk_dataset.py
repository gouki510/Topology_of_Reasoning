import random
from typing import Dict, Optional, Sequence
import torch
from torch.utils.data import IterableDataset
import transformers
from load_data.supervised_dataset import SupervisedDataset


class RandomWalkDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            tokenized (bool): If true we use a pretokenized dataset.
    """

    def __init__(
        self,
        graph_data,
        tokenizer,
        cot_length=10,
        seq_length=1024,
        continuous_steps=5,
        num_of_sequences=1024,
        chars_per_token=3.6,
    ):  
        super(RandomWalkDataset, self).__init__()
        self.data = graph_data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.cot_length = cot_length
        self.continuous_steps = continuous_steps
        self.epoch = 0
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.num_of_sequences = num_of_sequences
    
    def set_epoch(self, epoch):
        random.seed(epoch)

    def iter_fun(self):
        cot = []
        while True:
            if len(cot) >= self.cot_length or len(cot) == 0:
                print(cot)
                if len(cot) >= self.cot_length:
                    yield '\n'.join(cot) + '\n\n'
                cot = []
                node_id = random.randint(0, self.data.num_nodes-1)
                print(node_id)
            start_id = random.choice(self.data.graph[node_id])
            num_steps = random.randint(1, self.continuous_steps)
            end_id = min(start_id + num_steps - 1, len(self.data.cots) - 1)
            print(f"{start_id}: {end_id}")
            while self.data.example_ids[end_id] != self.data.example_ids[start_id]:
                end_id -= 1
            cot += list(self.data.cots[start_id: end_id + 1])
            node_id = self.data.node_ids[end_id]

    # def __len__(self):
    #     return len(self.data)//self.cot_length

    def __iter__(self):
        more_examples = True
        iterator = self.iter_fun()
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = self.iter_fun()
                    self.epoch += 1
                    print(f"Dataset epoch: {self.epoch}")
            
            # random.shuffle(buffer)
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.tokenizer.eos_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield dict(input_ids=torch.tensor(input_ids), labels=torch.tensor(input_ids))
                    

def make_random_walk_data_module(tokenizer: transformers.PreTrainedTokenizer, 
                            dataset, seq_length, cot_length, continuous_steps,
                            max_num_eval=1000, seed=42) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    random.seed(seed)
    train_dataset = RandomWalkDataset(dataset, tokenizer, cot_length,
        seq_length, continuous_steps)
    return dict(train_dataset=train_dataset, eval_dataset=None)