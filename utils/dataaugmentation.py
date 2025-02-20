import numpy as np
from Bio.Seq import Seq
import pandas as pd
import os

"""File per fare la data augmentation"""

class DNADataAugmenter:
    def __init__(self, data_dir='D:\\data', mutation_rate=0.01, window_shift=200):
        self.data_dir = data_dir
        self.mutation_rate = mutation_rate
        self.window_shift = window_shift

    def reverse_complement(self, sequence):
        """
        Genera il reverse complement
        """
        return str(Seq(sequence).reverse_complement())

    def introduce_mutations(self, sequence):
        """
        Serve per introdurre mutazioni casuali nelle sequenze
        """
        sequence = list(sequence)

        # Calcola il numero di mutazioni e poi le applica random nella sequenza

        n_mutations = int(len(sequence) * self.mutation_rate)
        positions = np.random.choice(len(sequence), n_mutations, replace=False)
        nucleotides = ['A', 'T', 'C', 'G']

        for pos in positions:
            if sequence[pos] != 'N':
                current = sequence[pos]
                options = [n for n in nucleotides if n != current]
                sequence[pos] = np.random.choice(options)

        return ''.join(sequence)


    def augment_sequence(self, sequence):
        """
        Applica tutte le tecniche di augmentation a una sequenza
        """
        augmented = []
        # prende la sequenza originale
        augmented.append(sequence)

        # fa reverse complement
        rc_seq = self.reverse_complement(sequence)
        augmented.append(rc_seq)

        # introduce mutazioni sulla sequenza originale e sul reverse complement
        augmented.append(self.introduce_mutations(sequence))
        augmented.append(self.introduce_mutations(rc_seq))

        return augmented

    def augment_dataset(self, df, output_prefix):
        """
        Augmenta l'intero dataset
        """
        print(f"Inizia augmentation per {output_prefix}")
        print(f"Dataset originale: {len(df)} sequenze")

        augmented_data = []

        for idx, row in df.iterrows():
            sequence = row['sequence']
            label = row['label']

            augmented_sequences = self.augment_sequence(sequence)

            for aug_seq in augmented_sequences:
                new_row = row.copy()
                new_row['sequence'] = aug_seq
                new_row['augmentation_type'] = 'augmented'
                augmented_data.append(new_row)

        # crea nuovo DataFrame con i dati augmented
        augmented_df = pd.DataFrame(augmented_data)

        # concatena originale e augmented
        final_df = pd.concat([df, augmented_df], ignore_index=True)

        print(f"Dataset augmentato: {len(final_df)} sequenze")
        print("\nDistribuzione labels:")
        print(final_df['label'].value_counts())

        output_file = os.path.join(self.data_dir, f'augmented_{output_prefix}.csv')
        final_df.to_csv(output_file, index=False)
        print(f"\nDataset augmentato salvato in: {output_file}")

        return final_df


if __name__ == "__main__":
    data_dir = 'D:\\data\\dati600'  # in dati600 ho i csv che intendo utilizzare per il mio task di classificazione
    augmenter = DNADataAugmenter(data_dir=data_dir)

    # Carica i dataset
    promoters_df = pd.read_csv(os.path.join(data_dir, 'standardized_promoters.csv'))
    enhancers_df = pd.read_csv(os.path.join(data_dir, 'standardized_enhancers.csv'))
    # Non carico gli introni perché ne ho già a sufficienza

    # aumenta il numero di promotori
    print("\nProcessing Promotori...")
    augmented_promoters = augmenter.augment_dataset(promoters_df, 'promoters')

    # aumenta il numero di enhancer
    print("\nProcessing Enhancer...")
    augmented_enhancers = augmenter.augment_dataset(enhancers_df, 'enhancers')

    target_size = 9015  # hard encoded perché so che è il numero minimo tra le dimensioni dopo l'augmentation

    # Lista delle classi
    classes = ['promoters', 'enhancers', 'introns']

    # Dizionario per salvare i df
    balanced_data = {}

    # Carica e campiona ogni dataset
    for class_name in classes:
        file_path = os.path.join(data_dir, f'augmented_{class_name}.csv')

        print(f"\nProcessing {class_name}...")
        print(f"Loading file: {file_path}")

        # Carica il dataset
        df = pd.read_csv(file_path)
        print(f"Original size: {len(df)} sequences")

        # Campiona 9000 sequenze
        df_sampled = df.sample(n=target_size, random_state=42)
        print(f"After sampling: {len(df_sampled)} sequences")

        # Salva nel dizionario
        balanced_data[class_name] = df_sampled

        # Salva il dataset campionato
        output_path = os.path.join(data_dir, f'balanced_{class_name}.csv')
        df_sampled.to_csv(output_path, index=False)
        print(f"Saved balanced dataset to: {output_path}")

    print("\nFinal Statistics:")
    for class_name, df in balanced_data.items():
        print(f"{class_name}: {len(df)} sequences")