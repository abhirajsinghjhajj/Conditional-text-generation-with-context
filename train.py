# train.py
import torch
import pandas as pd
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset
import os
import warnings

warnings.filterwarnings("ignore")

class SQuADDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenize input and target
        inputs = self.tokenizer(
            row['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            row['target_text'],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs.input_ids.flatten(),
            'attention_mask': inputs.attention_mask.flatten(),
            'labels': targets.input_ids.flatten()
        }

def train_model():
    # Setup directories
    os.makedirs('models', exist_ok=True)
    
    # Use MPS if available (optimized for your Mac)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Load datasets
    train_dataset = SQuADDataset('data/train.csv', tokenizer)
    val_dataset = SQuADDataset('data/val.csv', tokenizer)
    
    # Calculate steps per epoch for logging
    steps_per_epoch = len(train_dataset) // 8  # batch_size = 8
    
    # Training arguments - logging only at end of each epoch
    training_args = TrainingArguments(
        output_dir='models/t5-squad-finetuned',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=3e-4,
        logging_steps=steps_per_epoch,  # Log once per epoch
        logging_strategy='steps',
        eval_strategy='epoch',  # Evaluate after each epoch
        save_strategy='epoch',  # Save after each epoch
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        report_to=None,
        dataloader_pin_memory=False  # Avoid MPS pin memory warning
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("Starting training...")
    print("Training loss will be logged once per epoch")
    print("Evaluation will occur after each epoch")
    trainer.train()
    
    # Save final model
    trainer.save_model('models/t5-squad-final')
    tokenizer.save_pretrained('models/t5-squad-final')
    
    print("Training completed!")

if __name__ == "__main__":
    train_model()
