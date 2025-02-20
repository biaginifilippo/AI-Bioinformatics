import pandas as pd
from Bio import SeqIO
import gzip
import os
import requests

"""
In questo file vengono eseguite tutte quelle operazioni che si occupano di costruzione e standardizzazione delle 
sequenze all'interno del dataset, la data augmentation e il onehot encoding non sono qui 
"""
class SequenceFetcher:
    def __init__(self, data_dir='D:\\data'):
        self.data_dir = data_dir
        # URL base per le api
        self.ucsc_url = "https://api.genome.ucsc.edu/getData/sequence"

    def fetch_sequence_ucsc(self, chromosome, start, end):
        """
        Recupera la sequenza genomica da UCSC
        """
        params = {
            'genome': 'hg38',  # genoma umano di riferimento
            'chrom': chromosome,
            'start': start,
            'end': end
        }

        try:
            response = requests.get(self.ucsc_url, params=params)
            if response.ok:
                return response.json()['dna'].upper()
            else:
                print(f"Errore nel recupero della sequenza: {response.status_code}")
                return None
        except Exception as e:
            print(f"Errore nella richiesta: {e}")
            return None

    def process_enhancers(self, enhancers_df):
        """
        Recupera le sequenze per tutti gli enhancer
        """
        print("Inizia il recupero delle sequenze degli enhancer...")
        sequences = []

        for idx, row in enhancers_df.iterrows():
            if idx % 100 == 0:
                print(f"Processati {idx} enhancer...")

            sequence = self.fetch_sequence_ucsc(
                row['chromosome'],
                row['start'],
                row['end']
            )

            if sequence:
                sequences.append({
                    'id': row['id'],
                    'sequence': sequence,
                    'original_length': row['length'],
                    'label': row['label']
                })

        # crea nuovo df con le sequenze
        sequences_df = pd.DataFrame(sequences)

        # salva
        output_file = os.path.join(self.data_dir, 'enhancers_with_sequences.csv')
        sequences_df.to_csv(output_file, index=False)
        print(f"Sequenze salvate in: {output_file}")

        return sequences_df

class SequenceStandardizer:
    def __init__(self, data_dir='D:\\data', target_length=600):
        self.data_dir = data_dir
        self.target_length = target_length

    def standardize_sequence(self, sequence, current_length=None):
        """
        Standardizza una sequenza a 600 bp
        Se più lunga taglia dal centro
        """
        if current_length is None:
            current_length = len(sequence)
        # La sequenza viene tagliata prendendo la parte centrale
        if current_length > self.target_length:
            # Taglia dal centro
            excess = current_length - self.target_length
            start = excess // 2
            end = current_length - (excess - excess // 2)
            return sequence[start:end]
        else:
            return sequence



class EnhancerProcessor:
    def __init__(self, data_dir='D:\\data', min_length=1800, max_length=2200):
        self.data_dir = data_dir
        self.min_length = min_length
        self.max_length = max_length

    # vengono utilizzati 3 file per gli enhancer per aumentarne il numero
    def process_enhancer_files(self, file_names=['enhancer_main.txt', 'enhancer_gene.txt', 'enhancer_TF.txt']):
        """
        Processa multipli file di enhancer e li combina
        """
        print("1. Inizia processing dei file enhancer...")
        all_enhancers = pd.DataFrame()

        for file_name in file_names:
            file_path = os.path.join(self.data_dir, file_name)
            print(f"\nProcessing {file_name}...")

            try:
                # Leggi il file
                df = pd.read_csv(file_path, sep='\t')
                print(f"Lette {len(df)} righe da {file_name}")

                # filtra per specie umana (considera tutte le varianti di "human")
                human_mask = df['Species'].str.lower().str.contains('human', na=False)
                df = df[human_mask].copy()
                print(f"Dopo filtro specie: {len(df)} righe")

                # calcola lunghezza
                df['length'] = df['End_position'] - df['Start_position']

                # filtra per lunghezza
                mask = (df['length'] >= self.min_length) & (df['length'] <= self.max_length)
                df = df[mask].copy()
                print(f"Dopo filtro lunghezza: {len(df)} righe")

                # aggiungi al df
                if all_enhancers.empty:
                    all_enhancers = df
                else:
                    all_enhancers = pd.concat([all_enhancers, df], ignore_index=True)

            except Exception as e:
                print(f"Errore nel processing di {file_name}: {e}")
                continue

        # rimuovi duplicati
        all_enhancers = all_enhancers.drop_duplicates(
            subset=['Chromosome', 'Start_position', 'End_position']
        )

        # Crea df finale con tutti gli enhancer
        processed_df = pd.DataFrame({
            'id': all_enhancers['Enhancer_id'],
            'chromosome': all_enhancers['Chromosome'],
            'start': all_enhancers['Start_position'],
            'end': all_enhancers['End_position'],
            'length': all_enhancers['length'],
            'label': 'enhancer'
        })

        # stats finali
        print("\n2. Statistiche Finali:")
        print(f"Totale enhancer (dopo rimozione duplicati): {len(processed_df)}")
        print("\nDistribuzione lunghezze:")
        print(processed_df['length'].describe())
        print("\nDistribuzione per cromosoma:")
        print(processed_df['chromosome'].value_counts().head())

        output_file = os.path.join(self.data_dir, 'processed_enhancers.csv')
        processed_df.to_csv(output_file, index=False)
        print(f"\nDati salvati in: {output_file}")

        """
        Gli enhancers non vengono standardizzati qui perché hanno bisogno di essere fetchati,
        in questo df non ho le sequenze
        """

        return processed_df


class PromoterProcessor(SequenceStandardizer):
    def process_promoter_file(self, file_name='promoters.txt'):
        """
        Processa e standardizza i promotori
        """
        print("1. Inizia processing dei promotori...")
        file_path = os.path.join(self.data_dir, file_name)

        promoters = []
        try:
            with open(file_path, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    # Estrai la sequenza e standardizza
                    sequence = str(record.seq).upper()
                    std_sequence = self.standardize_sequence(sequence)

                    promoters.append({
                        'id': record.id,
                        'original_length': len(sequence),
                        'sequence': std_sequence,
                        'label': 'promoter'
                    })

            # crea df
            df = pd.DataFrame(promoters)
            print(f"\nProcessati {len(df)} promotori")

            # salva
            output_file = os.path.join(self.data_dir, 'standardized_promoters.csv')
            df.to_csv(output_file, index=False)
            print(f"Dati salvati in: {output_file}")

            return df

        except Exception as e:
            print(f"Errore nel processing dei promotori: {e}")
            return None


class IntronProcessor(SequenceStandardizer):
    def process_intron_file(self, file_name='gencode.v44.pc_transcripts.fa.gz'):
        """
        Processa e standardizza gli introni
        """
        print("1. Inizia processing degli introni...")
        file_path = os.path.join(self.data_dir, file_name)

        introni = []
        try:
            with gzip.open(file_path, "rt") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    sequence = str(record.seq).upper()

                    # Considera solo sequenze di lunghezza ragionevole
                    if 600 <= len(sequence) <= 800:
                        std_sequence = self.standardize_sequence(sequence)

                        introni.append({
                            'id': record.id,
                            'original_length': len(sequence),
                            'sequence': std_sequence,
                            'label': 'intron'
                        })

                    if len(introni) % 1000 == 0:
                        print(f"Processati {len(introni)} introni...")

            # df
            df = pd.DataFrame(introni)
            print(f"\nProcessati {len(df)} introni")

            # Prendi un campione casuale per bilanciare con gli altri dataset
            if len(df) > 10500:  # Assumiamo che vogliamo circa 2000-2500 esempi per classe
                df = df.sample(n=10500, random_state=42)
                print(f"Campionati {len(df)} introni per bilanciamento")

            output_file = os.path.join(self.data_dir, 'standardized_introns.csv')
            df.to_csv(output_file, index=False)
            print(f"Dati salvati in: {output_file}")

            return df

        except Exception as e:
            print(f"Errore nel processing degli introni: {e}")
            return None



def standardize_enhancer_sequences(df, target_length=600):
    """
    Standardizza le sequenze degli enhancer a 600 bp prendendo la parte centrale

    Args:
        df: DataFrame con colonna 'sequence'
        target_length: lunghezza target (default 600)
    """

    def cut_sequence_center(sequence):
        current_length = len(sequence)
        if current_length > target_length:
            # Taglia dal centro
            excess = current_length - target_length
            start = excess // 2
            end = current_length - (excess - excess // 2)
            return sequence[start:end]
        return sequence

    standardized_df = df.copy()

    # std
    print("Inizia standardizzazione sequenze...")
    standardized_df['sequence'] = standardized_df['sequence'].apply(cut_sequence_center)


    lengths = standardized_df['sequence'].str.len()
    print("\nStatistiche lunghezze dopo standardizzazione:")
    print(lengths.describe())

    return standardized_df


# Uso
# Assumendo che il tuo DataFrame si chiami enhancer_df:
# enhancer_standardized = standardize_enhancer_sequences(enhancer_df)
def process_all_sequences():
    """
    Processa e standardizza tutte le sequenze a 600bp, i promotir non subiscono nulla
    Mentre gli enhancer dopo essere stati scaricati dalle API e gli introni vengono
    Standardizzati tutti a lunghezza 600 e salvati in file che si chiamano
    standardized_promotors.csv standardazied_enhancers.csv e standardized_introns.csv
    """
    data_dir = 'D:\\data'

    # Processa e standardizza promotori
    print("\nProcessing Promotori...")
    promoter_processor = PromoterProcessor(data_dir)
    promoter_df = promoter_processor.process_promoter_file()

    # Processa e standardizza introni
    print("\nProcessing Introni...")
    intron_processor = IntronProcessor(data_dir)
    intron_df = intron_processor.process_intron_file()

    # Processa enhancer
    print("\nProcessing Enhancer...")
    enhancer_processor = EnhancerProcessor(data_dir)
    enhancer_df = enhancer_processor.process_enhancer_files()

    # Serve per reperire le sequenze degli enhancer con le API
    fetcher = SequenceFetcher()

    enhancers_with_sequences = fetcher.process_enhancers(enhancer_df)
    #Standaridzzo le sequenze
    enhancer_standardized = standardize_enhancer_sequences(enhancer_df)


    enhancer_standardized.to_csv('D:\\data\\standardized_enhancers.csv', index=False)
    print("\nStatistiche finali:")
    print(f"Enhancer processati: {len(enhancers_with_sequences)}")
    print("\nDistribuzione lunghezze:")
    print(enhancers_with_sequences['original_length'].describe())
    # Stampa statistiche finali
    print("\nStatistiche Finali del Dataset Standardizzato:")
    if promoter_df is not None:
        print(f"Promotori: {len(promoter_df)}")
    if intron_df is not None:
        print(f"Introni: {len(intron_df)}")
    if enhancer_df is not None:
        print(f"Enhancers: {len(promoter_df)}")



if __name__ == "__main__":
    process_all_sequences()
