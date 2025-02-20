import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import math
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from utils.dataset_dnabert import DNATokenizer, DNADataset, load_and_prepare_data

def plot_training_history(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
class DNATokenizer:
    def __init__(self):
        self.kmer_size = 3
        self.stride = 1
        # Creo il vocabolario per i 3-mer
        nucleotides = ['A', 'T', 'C', 'G']
        self.vocab = {}
        counter = 1  # 0 è riservato per il padding
        for i in nucleotides:
            for j in nucleotides:
                for k in nucleotides:
                    self.vocab[i + j + k] = counter
                    counter += 1

        self.vocab['[PAD]'] = 0
        self.vocab['[CLS]'] = len(self.vocab)
        self.vocab['[SEP]'] = len(self.vocab)

    def tokenize(self, sequence):
        # Converte la sequenza in kmers
        kmers = []
        for i in range(0, len(sequence) - self.kmer_size + 1, self.stride):
            kmer = sequence[i:i + self.kmer_size]
            kmers.append(self.vocab.get(kmer, self.vocab['[PAD]']))

        # Aggiungo [CLS] e [SEP]
        kmers = [self.vocab['[CLS]']] + kmers + [self.vocab['[SEP]']]
        return kmers


class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Tokenizzazione
        tokens = self.tokenizer.tokenize(sequence)

        # Padding/truncation
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.vocab['[PAD]']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        # Creazione della attention mask
        attention_mask = [1 if token != self.tokenizer.vocab['[PAD]'] else 0 for token in tokens]

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DNABERT(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super(DNABERT, self).__init__()
        self.num_classes = num_classes

        # Configurazione del modello
        self.hidden_size = 768
        self.num_layers = 12
        self.num_attention_heads = 12

        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(512, self.hidden_size)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_attention_heads,
            dim_feedforward=4 * self.hidden_size,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        # positional encoding
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # crea embedding
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # concatena positional encoding e embedding
        hidden_states = embeddings + position_embeddings

        hidden_states = hidden_states.transpose(0, 1)  # TransformerEncoder expects seq_len first
        transformer_output = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        transformer_output = transformer_output.transpose(0, 1)  # Batch first again

        # per la classificazione teniamo solo il token CLS
        sequence_output = transformer_output[:, 0]

        logits = self.classifier(sequence_output)

        return logits

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    progress_bar = tqdm(train_loader, desc='Training')

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader), predictions, true_labels


def validate(model, val_loader, criterion, device, desc="Validating"):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    progress_bar = tqdm(val_loader, desc=desc)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(val_loader), predictions, true_labels


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


if __name__ == "__main__":
    DATA_DIR = "D:/data/dati600"
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Crea cartella per i risultati
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/run_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # Inizializzazione
    tokenizer = DNATokenizer()

    # Carica tutti i dati
    all_sequences, _, all_labels, _ = load_and_prepare_data(DATA_DIR, test_size=0.0001)

    # Primo split: train e temp (80% - 20%)
    train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(
        all_sequences, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # Secondo split: validation e test dal temp (50% - 50%, risultando in 10% - 10% del totale)
    val_sequences, test_sequences, val_labels, test_labels = train_test_split(
        temp_sequences, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    # Crea dataset e dataloader
    train_dataset = DNADataset(train_sequences, train_labels, tokenizer)
    val_dataset = DNADataset(val_sequences, val_labels, tokenizer)
    test_dataset = DNADataset(test_sequences, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Inizializza il modello
    # model = LightDNABERT(num_classes=3, vocab_size=len(tokenizer.vocab))
    model = DNABERT(num_classes=3, vocab_size=len(tokenizer.vocab))
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    sys.exit(0)
    # Loss e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, 5, num_training_steps)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"Starting training on {DEVICE}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Training
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        # Validation
        val_loss, val_preds, val_labels = validate(
            model, val_loader, criterion, DEVICE
        )

        # Serve per plottare le loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Print
        print(f"\nTraining Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print("\nTraining Classification Report:")
        print(classification_report(train_labels, train_preds))
        print("\nValidation Classification Report:")
        print(classification_report(val_labels, val_preds))

        # Salavo i pesi se sono meglio dei migliori
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'{results_dir}/best_model.pt')

    # Plot
    plot_training_history(train_losses, val_losses, f'{results_dir}/training_history.png')

    # Cairco il migliore per il test
    print("\nPerforming final evaluation on test set...")
    best_model = DNABERT(num_classes=3, vocab_size=len(tokenizer.vocab))
    best_model.load_state_dict(torch.load(f'{results_dir}/best_model.pt')['model_state_dict'])
    best_model = best_model.to(DEVICE)


    test_loss, test_preds, test_labels = validate(
        best_model, test_loader, criterion, DEVICE, desc="Testing"
    )

    # Salva le metriche sul test
    with open(f'{results_dir}/test_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels, test_preds))

    # Plot
    plot_confusion_matrix(test_labels, test_preds, f'{results_dir}/confusion_matrix.png')