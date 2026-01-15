import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class Prepare_data:
    def __init__(self):
        # REMOVIDAS: Weight e IAT (99.9% correlacionadas com Number)
        self.selected_features = [
            'rst_count', 'Header_Length', 'Min', 'Variance', 
            # 'UDP', 
            'Covariance', 'TCP', 'Magnitue', 'Duration', 
            # 'Number',  # Mantemos apenas Number
            'Tot size'
        ]
        self.target = 'label'
        self.scaler = StandardScaler() 
        
        self.classes_mirai = [
            'BenignTraffic', 'Mirai-greeth_flood', 
            'Mirai-greip_flood', 'Mirai-udpplain'
        ]
    
    def clean_and_filter(self, df):
        # Remove infinitos e NaN
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Filtra apenas as classes de interesse
        df = df[df[self.target].isin(self.classes_mirai)]
        
        # REMOVE DUPLICATAS 
        df = df.drop_duplicates()
        
        return df
    
    def save_to_csv(self, input_path, output_path):
        """Lê o dataset gigante em partes e salva apenas o necessário"""
        if os.path.exists(output_path): 
            os.remove(output_path)
        
        cols_to_read = self.selected_features + [self.target]
        first_chunk = True
        
        # Lendo em blocos de 20k linhas para não travar a RAM
        chunks = pd.read_csv(input_path, chunksize=20000, usecols=cols_to_read)
        print(f" Filtrando dataset gigante: {input_path}...")
        
        for chunk in chunks:
            processed_chunk = self.clean_and_filter(chunk)
            if not processed_chunk.empty:
                processed_chunk.to_csv(output_path, mode='a', index=False, header=first_chunk)
                first_chunk = False
        
        print(f" Arquivo filtrado salvo em: {output_path}")
    
    def load_raw_data(self, path):
        """Retorna X e y crus para o processo de divisão"""
        df = pd.read_csv(path)
        
        # Remove duplicatas ao carregar também
        df = df.drop_duplicates()
        
        return df[self.selected_features], df[self.target]