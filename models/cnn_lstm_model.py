import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from utils.dataloader import prepare_data_loaders
from colorama import Fore

def get_stop_threshold_lstm(epoch, total_epochs):
    # Calcolare una soglia dinamica che decresce all'aumentare dell'epoca
    stop_threshold = 0.35 * (1 - epoch / total_epochs)**(5/3)

    # mi assicuro che la soglia non scenda mai sotto un valore minimo (es. 0.01)
    return max(stop_threshold, 0.01)

class CNNLSTMClassifier(nn.Module):
    def __init__(self,
                 input_channels=4,
                 conv_channels=[32, 64, 128],
                 kernel_sizes=[3, 5, 7],
                 lstm_hidden=256,
                 num_lstm_layers=2,
                 dropout=0.3,
                 num_classes=3):
        super().__init__()

        # Layer CNN
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        prev_channels = input_channels
        # fatto per non scrivere manualmente tutti layer di conv, si parte da 4 input channels e si va a
        # 32 con kernel size 3, poi da 32 a 64 con kernel di 5 e infine da 64 a 128 con kernel di 7, ognuno con padding
        for channels, kernel_size in zip(conv_channels, kernel_sizes):
            padding = kernel_size // 2
            self.conv_layers.append(nn.Conv1d(prev_channels, channels, kernel_size, padding=padding))
            self.batch_norms.append(nn.BatchNorm1d(channels))
            prev_channels = channels
        # Layer LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],  # 128
            hidden_size=lstm_hidden,  # 256
            num_layers=num_lstm_layers,  # 2
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Layer di classificazione
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.layer_norm = nn.LayerNorm(lstm_hidden)
        self.fc2 = nn.Linear(lstm_hidden, num_classes)

        # Inizializzazione pesi
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # Feature extraction CNN
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = F.relu(bn(conv(x)))

        # scambio canali con lunghezza sequenza per adattarlo all'input che si aspetta la lstm
        x = x.transpose(1, 2)

        # Processing LSTM
        lstm_out, _ = self.lstm(x)

        # si toglie la sequenza perché le info le ho già processato e sono dentro all'ultima dimensione
        lstm_out = lstm_out[:, -1, :]

        # Classificazione
        x = F.relu(self.fc1(lstm_out))
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class CNNLSTMClassifierUni(nn.Module):
    def __init__(self,
                 input_channels=4,
                 conv_channels=[32, 64, 128],
                 kernel_sizes=[3, 5, 7],
                 lstm_hidden=256,
                 num_lstm_layers=2,
                 dropout=0.3,
                 num_classes=3):
        super().__init__()

        # Layer CNN
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        prev_channels = input_channels
        # fatto per non scrivere manualmente tutti layer di conv, si parte da 4 input channels e si va a
        # 32 con kernel size 3, poi da 32 a 64 con kernel di 5 e infine da 64 a 128 con kernel di 7, ognuno con padding
        for channels, kernel_size in zip(conv_channels, kernel_sizes):
            padding = kernel_size // 2
            self.conv_layers.append(nn.Conv1d(prev_channels, channels, kernel_size, padding=padding))
            self.batch_norms.append(nn.BatchNorm1d(channels))
            prev_channels = channels
        # Layer LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],  # 128
            hidden_size=lstm_hidden,  # 256
            num_layers=num_lstm_layers,  # 2
            batch_first=True,
            bidirectional=False,  # Versione unidirezionale visto che nel dataset è già presente il reverse complement
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Layer di classificazione
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden, lstm_hidden//2)
        self.layer_norm = nn.LayerNorm(lstm_hidden//2)
        self.fc2 = nn.Linear(lstm_hidden//2, num_classes)

        # Inizializzazione pesi
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # Feature extraction CNN
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = F.relu(bn(conv(x)))

        # scambio canali con lunghezza sequenza per adattarlo all'input che si aspetta la lstm
        x = x.transpose(1, 2)

        # Processing LSTM
        lstm_out, _ = self.lstm(x)

        # si toglie la sequenza perché le info le ho già processato e sono dentro all'ultima dimensione
        lstm_out = lstm_out[:, -1, :]

        # Classificazione finale
        x = F.relu(self.fc1(lstm_out))
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


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


def plot_training_history_lstm(history):
    # Creiamo una nuova figura con due subplot affiancati
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot della loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    # Plot dell'accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.savefig('cnn_lstm_acc_loss.png', bbox_inches='tight', dpi=300)

    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_confusion_matrix_lstm(true_labels, predictions, classes=['Promoter', 'Enhancer', 'Intron']):


    # Crea la confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.savefig('cnn_lstm_confusion_matrix.png', bbox_inches='tight', dpi=300)

    plt.show(block=False)
    plt.pause(1)
    plt.close()

def train_cnn_lstm(model, train_loader, val_loader, test_loader, device, num_epochs=150, patience=5):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.00055, weight_decay=0.007)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     patience=5,
                                                     factor=0.8,
                                                     min_lr=1e-7)

    model = model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    start_time = time.time()

    print(f"\nTraining on {Fore.GREEN}{torch.cuda.get_device_name(device)}{Fore.RESET}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    warning = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss / len(train_loader):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nTrain Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), '../cnn_lstm/dna_cnn_lstm.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n{Fore.YELLOW}Early stopping triggered!{Fore.RESET}")
                break

        if val_loss > train_loss + get_stop_threshold_lstm(epoch, num_epochs):
            if epoch > 80:
                if warning < 1:
                    warning += 1
                    print(f"{Fore.YELLOW}warning triggered{Fore.RESET}")
                    continue
                else:
                   print(f"{Fore.YELLOW}early stop triggered for diverging losses{Fore.RESET}")
                   break
            else:
                warning = 0

    training_time = time.time() - start_time
    print(f"{Fore.BLUE}\nTraining completed in {training_time / 60:.2f} minutes")

    model.load_state_dict(best_model_state)

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader,
                                                            criterion, device)
    print(f"{Fore.BLUE}\nTest Set Performance:{Fore.RESET}")
    print(f"{Fore.BLUE}Loss: {test_loss:.4f} - Accuracy: {test_acc * 100:.2f}%{Fore.RESET}")
    print(f"{Fore.BLUE}\nClassification Report:{Fore.RESET}")
    print(classification_report(test_labels, test_preds,
                                target_names=['Promoter', 'Enhancer', 'Intron']))

    plot_training_history_lstm(history)
    plot_confusion_matrix_lstm(test_labels, test_preds)

    return model, history

if __name__ == "__main__":
    data_dir = "D:\\data\\dati600"
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = prepare_data_loaders(data_dir, batch_size)
    model = CNNLSTMClassifierUni()

    model, history = train_cnn_lstm(model, train_loader, val_loader, test_loader, device)
    torch.save(model.state_dict(), '../trash can/dna_cnn_lstm.pth')