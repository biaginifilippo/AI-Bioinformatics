import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from DNABERT import DNABERT
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from utils.dataset_dnabert import DNADataset, load_and_prepare_data
from utils.dataset_dnabert import DNATokenizer6mer as DNATokenizer


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

class LightDNABERT(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super(LightDNABERT, self).__init__()
        self.num_classes = num_classes

        # Dimensioni del modello
        self.hidden_size = 256
        self.num_layers = 4
        self.num_attention_heads = 4

        # Embedding layers
        self.embeddings = nn.Embedding(vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(600, self.hidden_size)

        # Layer Normalization e Dropout
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_attention_heads,
            dim_feedforward=2 * self.hidden_size,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        hidden_states = embeddings + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))

        hidden_states = hidden_states.transpose(0, 1)
        transformer_output = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        transformer_output = transformer_output.transpose(0, 1)

        sequence_output = transformer_output[:, 0]
        logits = self.classifier(sequence_output)

        return logits


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    # Aggiungi tqdm
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

        # Aggiorna la progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader), predictions, true_labels


def validate(model, val_loader, criterion, device, desc="Validating"):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    # Aggiungi tqdm anche per la validation
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


if __name__ == "__main__":
    DATA_DIR = "D:/data/dati600"
    BATCH_SIZE = 256
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
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
    model = LightDNABERT(num_classes=3, vocab_size=len(tokenizer.vocab))
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    # Loss e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

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

        # Append losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print
        print(f"\nTraining Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print("\nTraining Classification Report:")
        print(classification_report(train_labels, train_preds))
        print("\nValidation Classification Report:")
        print(classification_report(val_labels, val_preds))

        # Salva se è il migliore
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

    # Carica il modello migliore per fare il testset
    print("\nPerforming final evaluation on test set...")
    best_model = LightDNABERT(num_classes=3, vocab_size=len(tokenizer.vocab))
    best_model.load_state_dict(torch.load(f'{results_dir}/best_model.pt')['model_state_dict'])
    best_model = best_model.to(DEVICE)

    test_loss, test_preds, test_labels = validate(
        best_model, test_loader, criterion, DEVICE, desc="Testing"
    )

    with open(f'{results_dir}/test_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels, test_preds))

    # Plot
    plot_confusion_matrix(test_labels, test_preds, f'{results_dir}/confusion_matrix.png')