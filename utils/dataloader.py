import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

"""
File contentente il codice per costruire i dataset e i dataloader
"""

class DNASequenceDataset(Dataset):
    def __init__(self, data_dir):
        # Carica i tre dataset bilanciati
        promoters = pd.read_csv(f"{data_dir}/balanced_promoters.csv")
        enhancers = pd.read_csv(f"{data_dir}/balanced_enhancers.csv")
        introns = pd.read_csv(f"{data_dir}/balanced_introns.csv")

        # Aggiungi le label
        promoters['label'] = 0  # Promoters
        enhancers['label'] = 1  # Enhancers
        introns['label'] = 2  # Introns

        self.data = pd.concat([promoters, enhancers, introns], ignore_index=True)

        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Dizionario per la codifica one-hot dei nucleotidi
        self.nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['sequence']
        label = self.data.iloc[idx]['label']

        # Converti la sequenza in one-hot encoding
        one_hot = torch.zeros((4, 600))  # 4 nucleotidi (ACGT)x 600bp
        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.nucleotide_map:  # Se è ACGT
                one_hot[self.nucleotide_map[nucleotide], i] = 1
            # Se è N lascia tutti 0 in quella posizione

        return one_hot, torch.tensor(label, dtype=torch.long)


def prepare_data_loaders(data_dir, batch_size=32):
    """
    Prepara i data loader con split 80-10-10
    """
    # Crea il dataset completo
    dataset = DNASequenceDataset(data_dir)

    # Calcola le dimensioni degli split
    total_size = len(dataset)
    test_size = int(total_size * 0.1)
    val_size = int(total_size * 0.1)
    train_size = total_size - test_size - val_size

    # Split del dataset
    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Crea i data loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader


