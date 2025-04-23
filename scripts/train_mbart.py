import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, MBartTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import evaluate
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128, src_lang="en_XX", tgt_lang="fr_XX"):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data.iloc[idx]['en']
        target_text = self.data.iloc[idx]['fr']

        # Tokenize inputs
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize targets
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = source_encoding['input_ids'].squeeze()
        attention_mask = source_encoding['attention_mask'].squeeze()
        labels = target_encoding['input_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def train_model(model, train_dataloader, val_dataloader, optimizer, device, num_epochs=3):
    model.train()
    best_val_loss = float('inf')
    metric = evaluate.load("sacrebleu")

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

        avg_train_loss = total_loss / len(train_dataloader)
        val_loss = evaluate_model(model, val_dataloader, device, metric)
        
        logger.info(f"Epoch {epoch + 1}: Average training loss = {avg_train_loss:.3f}")
        logger.info(f"Epoch {epoch + 1}: Validation loss = {val_loss:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join('models', 'checkpoints', 'best_model'))
            logger.info("Saved new best model")

def evaluate_model(model, dataloader, device, metric):
    model.eval()
    total_loss = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                no_repeat_ngram_size=2
            )

            # Decode predictions and references
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_refs)

    # Calculate BLEU score
    bleu_score = metric.compute(predictions=predictions, references=references)
    logger.info(f"BLEU score: {bleu_score['score']:.2f}")

    return total_loss / len(dataloader)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "fr_XX"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Load datasets
    train_df = pd.read_csv('../data/processed/train.csv')
    val_df = pd.read_csv('../data/processed/val.csv')
    
    # Create datasets
    train_dataset = TranslationDataset(train_df, tokenizer)
    val_dataset = TranslationDataset(val_df, tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Train the model
    train_model(model, train_dataloader, val_dataloader, optimizer, device)

if __name__ == "__main__":
    main()