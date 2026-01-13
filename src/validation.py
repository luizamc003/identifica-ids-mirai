import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

class ModelValidator:
    def __init__(self, model_path="results/xgb_mirai_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
        
        # Carrega o dicionário que salvamos com modelo e encoder
        data = joblib.load(model_path)
        self.model = data['model']
        self.encoder = data['encoder']
        print(f"Modelo e Encoder carregados com sucesso!")

    def run_stress_test(self, test_csv_path, sample_size=50000):
        print(f"Iniciando teste de stress em: {test_csv_path}")
        
        # 1. Carrega o combo salvo (Modelo + Encoder + Scaler)
        # IMPORTANTE: Você precisa ter salvado o scaler no joblib anteriormente!
        data = joblib.load("results/xgb_mirai_model.pkl")
        model = data['model']
        encoder = data['encoder']
        scaler = data['scaler'] # <--- VOCÊ PRECISA DISSO AQUI
        
        df_new = pd.read_csv(test_csv_path).sample(n=sample_size, random_state=99)
        
        features = [
            'rst_count', 'Header_Length', 'Min', 'Variance', 'UDP', 
            'Covariance', 'TCP', 'Magnitue', 'Duration', 
            'Number', 'Weight', 'Tot size', 'IAT'
        ]
        
        X_raw = df_new[features]
        y_true_labels = df_new['label']
        
        # --- AQUI ESTAVA O ERRO ---
        # 2. Normaliza os novos dados usando o MESMO scaler do treino
        X_test_scaled = scaler.transform(X_raw) 
        
        # 3. Converte labels
        y_true = encoder.transform(y_true_labels)
        
        # 4. Predição usando os dados ESCALONADOS
        y_pred = model.predict(X_test_scaled)
        
        # 5. Relatório Técnico
        self._generate_results(y_true, y_pred)

    def _generate_results(self, y_true, y_pred):
        print("\n" + "="*40)
        print("RELATÓRIO TÉCNICO DE VALIDAÇÃO EXTERNA")
        print("="*40)
        
        report = classification_report(y_true, y_pred, target_names=self.encoder.classes_)
        print(report)
        
        # Salva o relatório de stress para comparar com o de treino
        with open("results/relatorio_stress_test.txt", "w") as f:
            f.write(report)
            
        # Gera a Matriz de Confusão de Validação
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, 
            display_labels=self.encoder.classes_, 
            cmap='OrRd', # Cor laranja/vermelha para diferenciar do treino
            xticks_rotation=45,
            ax=ax
        )
        plt.title("Matriz de Confusão: Validação com Dados Desconhecidos")
        plt.tight_layout()
        plt.savefig("results/matriz_stress_test.png")
        print(f"\nMatriz de confusão salva em: results/matriz_stress_test.png")
        plt.show()