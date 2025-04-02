import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


def load_sst2_data(tokenizer, batch_size, eval_batch_size, rank, world_size):
    """Load SST-2 dataset for sentiment classification"""
    train_dataset = load_dataset("glue", "sst2", split="train")
    eval_dataset = load_dataset("glue", "sst2", split="validation")
    
    block_size = 128
    
    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=block_size)
        
    # Preprocess train and validation sets
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Create distributed samplers and dataloaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
    
    return train_dataloader, eval_dataloader

def load_c4_data(tokenizer, batch_size, rank, world_size):
    """Load C4-en dataset for language modeling"""
    # Load only training data, no validation set
    train_dataset = load_dataset("c4", "en", split="train", streaming=True)
    
    block_size = 2048
    
    def tokenize_function(data):
        outputs = tokenizer(
            data["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )
        return outputs
    
    # Preprocess train dataset and remove unnecessary columns
    tokenized_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text", "timestamp", "url"]
    )
    
    tokenized_dataset.set_format(type="torch")
    
    # Create distributed sampler and dataloader
    train_sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(tokenized_dataset, sampler=train_sampler, batch_size=batch_size)
    
    # Return train dataloader and None for eval_dataloader, as we'll compute ppl differently
    return train_dataloader, None

def load_data_and_model(dataset_name, model_name, batch_size, eval_batch_size, rank, world_size):
    """
    Load dataset and model.
    Return Trainloader, Evalloader, tokenizer, model objects.
    """
    if dataset_name == "sst2" and model_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        train_dataloader, eval_dataloader = load_sst2_data(tokenizer, batch_size, eval_batch_size, rank, world_size)
        return train_dataloader, eval_dataloader, tokenizer, model, 'classification'
        
    elif dataset_name == "c4en":
        if model_name == "llama150m":
            model_name = "PrimeIntellect/llama-150m-fresh"
        elif model_name == 'llama1b':
            model_name = "PrimeIntellect/llama-1b-fresh"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        train_dataloader, eval_dataloader = load_c4_data(tokenizer, batch_size, rank, world_size)
        return train_dataloader, None, tokenizer, model, 'language_modeling'
        
    else:
        raise ValueError(f"Unsupported combination of dataset and model: {dataset_name}, {model_name}")
    
