import os
os.environ['HF_HOME']='/data/hfhub'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import time
import argparse
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim import SGD
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import numpy as np

def load_and_prepare_data(tokenizer, batch_size, rank, world_size):
    """
    Load and prepare the dataset for Hogwild training.
    """
    # Load dataset
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"]
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    
    # For Hogwild, we'll create splits without using DistributedSampler
    # Each process gets its own subset of data
    indices = np.arange(len(tokenized_dataset))
    np.random.shuffle(indices)
    
    # Split indices for each process
    per_worker = len(indices) // world_size
    start_idx = rank * per_worker
    end_idx = start_idx + per_worker if rank < world_size - 1 else len(indices)
    
    process_indices = indices[start_idx:end_idx]
    process_dataset = torch.utils.data.Subset(tokenized_dataset, process_indices)
    
    # Create dataloader without sampler (each process has its dedicated data)
    train_dataloader = DataLoader(
        process_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda data: {
            'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in data]),
            'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in data]),
            'labels': torch.tensor([x['label'] for x in data])
        }
    )
    
    return train_dataloader, len(tokenized_dataset)

def train_worker(rank, shared_model, args):
    """
    Worker process for Hogwild training.
    """
    # Set the device
    device = torch.device(f"cuda:{rank}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create a process-local model that shares memory with the global model
    local_model = shared_model.to(device)
    
    # Prepare data
    train_dataloader, dataset_size = load_and_prepare_data(
        tokenizer, args.batch_size, rank, args.world_size
    )
    
    # Initialize optimizer - SGD is typically used with Hogwild
    # Using higher learning rate as it's recommended for asynchronous SGD
    optimizer = SGD(local_model.parameters(), lr=args.learning_rate * 10)
    
    # Training loop
    if rank == 0:
        print(f"Starting Hogwild training on {args.world_size} GPUs...")
    
    for epoch in range(args.num_epochs):
        # Reset timing stats
        epoch_start_time = time.time()
        batch_times = []
        
        # Training loop
        local_model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            batch_start_time = time.time()
            
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = local_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update parameters immediately without synchronization (Hogwild approach)
            optimizer.step()
            
            # Calculate timing
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            total_loss += loss.item()
            
            # Log progress
            if rank == 0 and (step + 1) % args.logging_steps == 0:
                avg_batch_time = sum(batch_times[-args.logging_steps:]) / min(args.logging_steps, len(batch_times))
                print(f"Epoch: {epoch+1}/{args.num_epochs}, Step: {step+1}/{len(train_dataloader)}, "
                      f"Loss: {loss.item():.4f}, Batch Time: {avg_batch_time:.4f}s")
        
        # End of epoch stats
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        avg_loss = total_loss / len(train_dataloader)
        
        if rank == 0:
            avg_batch_time = sum(batch_times) / len(batch_times)
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - Process {rank} Avg. Loss: {avg_loss:.4f}, "
                  f"Avg. Batch Time: {avg_batch_time:.4f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=100)
    
    args = parser.parse_args()
    
    # Use all available GPUs (up to 4)
    args.world_size = torch.cuda.device_count()
    assert args.world_size <= 4, f"This script is configured for up to 4 GPUs, but {args.world_size} were found."
    
    # Initialize a shared model
    shared_model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    # Share memory across processes
    shared_model.share_memory()
    
    # Start multiprocessing
    mp.set_start_method('spawn', force=True)
    processes = []
    
    # Create and start processes
    for rank in range(args.world_size):
        p = mp.Process(target=train_worker, args=(rank, shared_model, args))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print("Hogwild training completed!")

if __name__ == "__main__":
    main()