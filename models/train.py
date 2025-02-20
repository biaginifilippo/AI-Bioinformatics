import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from utils.dataloader import prepare_data_loaders
# from models.model3 import DeePromoterModel
from models.lightModel3 import DeePromoterModel

""" Train loop per model3 """


def get_stop_threshold(epoch, total_epochs):
    # Calcolare una soglia dinamica che decresce all'aumentare dell'epoca
    stop_threshold = 0.7 * (1 - epoch / total_epochs) ** (4 / 3)
    # stop_threshold = 0.25 * (1 - epoch / total_epochs)**(2/3)

    # Assicurarsi che la soglia non scenda mai sotto un valore minimo (es. 0.01)
    return max(stop_threshold, 0.01)
"""
Questo file traina model3

"""
#TODO aggiustare gli iperparametri come learning rate, weight decay, factor nello scheduler, patience dello scheduler
# provare ad allenare model 3 con batch di dimensione differente da 2048, potrebbe distruggere gli altri perché
# è avvantaggiato dalla leggerezza di potersi allenare con batch enormi
#
# def train_epoch(model, train_loader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     # Usa tqdm per la progress bar
#     pbar = tqdm(train_loader, desc='Training')
#
#     for inputs, labels in pbar:
#         inputs, labels = inputs.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
#
#         # Aggiorna la progress bar
#         pbar.set_postfix({'loss': f'{running_loss / total:.4f}',
#                           'acc': f'{100. * correct / total:.2f}%'})
#
#     return running_loss / len(train_loader), correct / total


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(data_loader), correct / total, all_preds, all_labels


def plot_confusion_matrix(true_labels, predictions, classes=['Promoter', 'Enhancer', 'Intron']):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.show()


def get_warmup_linear_schedule(optimizer, num_epochs, num_training_steps_per_epoch):
    """
    Crea uno scheduler che:
    1. Aumenta linearmente il learning rate da 0 al valore base nelle prime num_epochs/10 epoche (fase di warmup)
    2. Diminuisce linearmente il learning rate fino a 0 nelle epoche rimanenti

    Args:
        optimizer: L'ottimizzatore pytorch per cui creare lo scheduler
        num_epochs: Numero totale di epoche di training
        num_training_steps_per_epoch: Numero di batch per epoca

    Returns:
        LambdaLR scheduler
    """
    # Calcolo del numero totale di step di training
    total_training_steps = num_epochs * num_training_steps_per_epoch

    warmup_steps = (num_epochs // 5) * num_training_steps_per_epoch

    def lr_lambda(current_step):
        # Durante la fase di warmup
        if current_step < warmup_steps:
            # Aumenta linearmente da 0 a 1
            return float(current_step) / float(max(1, warmup_steps))

        # Dopo il warmup, decresce linearmente fino a 0
        return max(
            0.0,
            float(total_training_steps - current_step) /
            float(max(1, total_training_steps - warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)


def train_model(model, train_loader, val_loader, test_loader, device,
                num_epochs=100, patience=8):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = total_steps // 5
    warning = 0
    # Scheduler con warmup
    scheduler = get_warmup_linear_schedule(optimizer, num_epochs, 11)
    model = model.to(device)
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Training on {device}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("Starting training...")

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping perché aiutava a stabilizzare il train
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            current_lr = scheduler.get_last_lr()[0]

            pbar.set_postfix({'loss': f'{running_loss / total:.4f}',
                              'acc': f'{100. * correct / total:.2f}%',
                             'lr': f'{current_lr:.2e}'})

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader,
                                                            criterion, device)


        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), f'model3_weights\\dna_classifier_checkpoint{epoch}_2048_lr_schedulato.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
        if val_loss > train_loss + get_stop_threshold(epoch, num_epochs):
            if warning < 1:
                warning = 1
                print(f'warning triggered: {warning}')
                continue
            print(f"Early stop triggered at epoch {epoch}, diverging losses: {train_loss} vs {val_loss}")
            break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.2f} minutes")

    model.load_state_dict(best_model_state)

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader,
                                                            criterion, device)
    print("\nTest Set Performance:")
    print(f"Loss: {test_loss:.4f} - Accuracy: {test_acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=['Promoter', 'Enhancer', 'Intron']))

    plot_training_history(history)
    plot_confusion_matrix(test_labels, test_preds)

    return model, history


if __name__ == "__main__":
    # Parametri
    data_dir = "D:\\data\\dati600"
    batch_size = 2048
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepara i data loader
    train_loader, val_loader, test_loader = prepare_data_loaders(data_dir, batch_size)

    # Crea il modello
    model = DeePromoterModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    sys.exit(0)
    # Training
    model, history = train_model(model, train_loader, val_loader, test_loader, device, num_epochs=num_epochs)

    # Salva i pesi
    torch.save(model.state_dict(), 'dna_classifier_try_2048_lr_schedulato.pth')