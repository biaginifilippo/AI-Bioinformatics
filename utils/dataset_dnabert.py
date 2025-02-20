import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

class DNATokenizer6mer:
    def __init__(self):
        self.kmer_size = 6
        self.stride = 1
        nucleotides = ['A', 'T', 'C', 'G']
        self.vocab = {}
        counter = 1
        # Viene costruito un dictionayr per ogni possibile tripletta, quindi si generano tutte le possibili
        # permutazioni e si assegna un numero crescente a ognuna di queste, la prima ha 1 l'ultima ha 64 (4x4x4)
        for i in nucleotides:
            for j in nucleotides:
                for k in nucleotides:
                    for l in nucleotides:
                        for m in nucleotides:
                            for n in nucleotides:
                                self.vocab[i + j + k + l + m + n] = counter
                                counter += 1

        self.vocab['[PAD]'] = 0  # aggiungo il token di padding nel dictionary = 0
        self.vocab['[CLS]'] = len(self.vocab)  # 65
        self.vocab['[SEP]'] = len(self.vocab)  # 66

    def tokenize(self, sequence):
        kmers = []
        for i in range(0, len(sequence) - self.kmer_size + 1, self.stride):
            kmer = sequence[i:i + self.kmer_size]
            kmers.append(self.vocab.get(kmer, self.vocab['[PAD]']))

        kmers = [self.vocab['[CLS]']] + kmers + [self.vocab['[SEP]']]
        return kmers

class DNATokenizer:
    def __init__(self):
        self.kmer_size = 3
        self.stride = 1
        nucleotides = ['A', 'T', 'C', 'G']
        self.vocab = {}
        counter = 1
        # Viene costruito un dictionayr per ogni possibile tripletta, quindi si generano tutte le possibili
        # permutazioni e si assegna un numero crescente a ognuna di queste, la prima ha 1 l'ultima ha 64 (4x4x4)
        for i in nucleotides:
            for j in nucleotides:
                for k in nucleotides:
                    self.vocab[i + j + k] = counter
                    counter += 1

        self.vocab['[PAD]'] = 0  # aggiungo il token di padding nel dictionary = 0
        self.vocab['[CLS]'] = len(self.vocab)  # 65
        self.vocab['[SEP]'] = len(self.vocab)  # 66

    def tokenize(self, sequence):
        kmers = []
        for i in range(0, len(sequence) - self.kmer_size + 1, self.stride):
            kmer = sequence[i:i + self.kmer_size]
            kmers.append(self.vocab.get(kmer, self.vocab['[PAD]']))

        kmers = [self.vocab['[CLS]']] + kmers + [self.vocab['[SEP]']]
        return kmers


class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=600):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        tokens = self.tokenizer.tokenize(sequence)

        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.vocab['[PAD]']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        attention_mask = [1 if token != self.tokenizer.vocab['[PAD]'] else 0 for token in tokens]

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_prepare_data(data_dir, test_size=0.1, random_state=42):
    """
    Carica e prepara i dati dai file CSV
    """
    enhancers = pd.read_csv(f"{data_dir}/balanced_enhancers.csv")
    introns = pd.read_csv(f"{data_dir}/balanced_introns.csv")
    promoters = pd.read_csv(f"{data_dir}/balanced_promoters.csv")

    sequences = []
    labels = []

    # Enhancers - label 0
    sequences.extend(enhancers['sequence'].values)
    labels.extend([0] * len(enhancers))

    # Introns - label 1
    sequences.extend(introns['sequence'].values)
    labels.extend([1] * len(introns))

    # Promoters - label 2
    sequences.extend(promoters['sequence'].values)
    labels.extend([2] * len(promoters))

    # Split
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    return train_sequences, val_sequences, train_labels, val_labels