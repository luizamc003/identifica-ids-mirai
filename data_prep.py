import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Prepare_data:
    def __init__(self):
        # As 13 colunas selecionadas via análise de correlação 
        self.selected_features = [
            'rst_count',      # rst_count
            'Header_Length',  # Header_Length
            'Min',             # Min
            'Variance',       # Variance
            'UDP',            # Protocolo UDP (binário no seu CSV)
            'Covariance',     # Covariance
            'TCP',            # Protocolo TCP (binário no seu CSV)
            'Magnitue',       # Nota: No seu CSV está escrito 'Magnitue' (sem o 'n')
            'Duration',       # Duration
            'Number',         # Number
            'Weight',         # Weight
            'Tot size',       # No seu CSV é 'Tot size' (com espaço)
            'IAT'             # IAT
        ]
        self.target = 'label'
        self.scaler = StandardScaler() 
        
        # Mapeamento das classes de interesse (Benigno e variantes Mirai)
        self.classes_mirai = [
            'BenignTraffic', 
            'Mirai-greeth_flood', 
            'Mirai-greip_flood', 
            'Mirai-udpplain'
        ]

    def clean_and_filter(self, df):
        # Remove valores infinitos e nulos (comuns em cálculos de fluxo) 
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Filtra apenas as classes que o estudo validou 
        df = df[df[self.target].isin(self.classes_mirai)]
        return df

    def save_to_csv(self, input_path, output_path):
        first_chunk = True
        # Usamos apenas as colunas que existem no seu arquivo
        cols_to_read = self.selected_features + [self.target]
        
        # O pandas vai ler apenas o que for útil
        chunks = pd.read_csv(input_path, chunksize=100000, usecols=cols_to_read)
        
        for chunk in chunks:
            processed_chunk = self.clean_and_filter(chunk)
            if not processed_chunk.empty:
                processed_chunk.to_csv(output_path, mode='a', index=False, header=first_chunk)
                first_chunk = False

    def transform(self, processed_csv_path):
        """
        Lê o arquivo já filtrado e aplica a normalização (StandardScaler)
        """
        df = pd.read_csv(processed_csv_path)
        X = df[self.selected_features]
        y = df[self.target]
        
        # O StandardScaler é aplicado aqui para preparar para o modelo
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y