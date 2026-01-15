import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

class ModelValidator:
    def __init__(self, model_path="results/xgb_mirai_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
        
        self.model_path = model_path
        
        # Carrega o dicionário completo
        data = joblib.load(model_path)
        self.model = data['model']
        self.encoder = data['encoder']
        self.scaler = data['scaler']
        self.features = data.get('features', None)
        
        print("="*60)
        print("MODELO CARREGADO COM SUCESSO")
        print("="*60)
        print(f" Modelo: XGBoost")
        print(f" Classes: {list(self.encoder.classes_)}")
        if self.features:
            print(f" Features ({len(self.features)}): {self.features}")
        print("="*60)
    
    def run_stress_test(self, dataset_path, sample_size=None):
        """
        Testa o modelo em dados completamente novos
        
        dataset_path: caminho do CSV de validação
        sample_size: quantas amostras usar (None = todas)
        """
        print("\n" + "="*60)
        print("INICIANDO TESTE DE STRESS (VALIDAÇÃO EXTERNA)")
        print("="*60)
        print(f"Dataset: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")
        
        # Carrega o dataset de validação
        df_val = pd.read_csv(dataset_path)
        
        # Se sample_size for especificado, faz amostragem
        if sample_size and sample_size < len(df_val):
            print(f" Amostrando {sample_size} de {len(df_val)} amostras...")
            df_val = df_val.sample(n=sample_size, random_state=99)
        else:
            print(f" Usando todas as {len(df_val)} amostras")
        
        # IMPORTANTE: Usar as mesmas features que o modelo foi treinado
        # Features SEM Weight e IAT (removidas por correlação)
        features = [
            'rst_count', 'Header_Length', 'Min', 'Variance', 'UDP', 
            'Covariance', 'TCP', 'Magnitue', 'Duration', 
            'Number', 'Tot size'
        ]
        
        # Verifica se todas as features existem no dataset
        missing_features = [f for f in features if f not in df_val.columns]
        if missing_features:
            raise ValueError(f"Features faltando no dataset: {missing_features}")
        
        # Extrai features e labels
        X_raw = df_val[features]
        y_true_labels = df_val['label']
        
        print(f"\n Distribuição das classes no dataset de validação:")
        print(y_true_labels.value_counts())
        
        # --- PIPELINE DE PROCESSAMENTO ---
        
        # 1. Escala os dados usando o MESMO scaler do treino
        X_scaled = self.scaler.transform(X_raw)
        
        # 2. Codifica os labels
        y_true = self.encoder.transform(y_true_labels)
        
        # 3. Faz predições
        y_pred = self.model.predict(X_scaled)
        
        # 4. Calcula acurácia geral
        acc = accuracy_score(y_true, y_pred)
        
        # 5. Gera relatórios
        self._generate_results(y_true, y_pred, acc)
        
        print("\n" + "="*60)
        print("TESTE DE STRESS CONCLUÍDO")
        print("="*60)
    
    def _generate_results(self, y_true, y_pred, accuracy):
        """Gera e salva relatórios de validação"""
        
        # Relatório de classificação
        print("\n" + "="*60)
        print("RELATÓRIO DE VALIDAÇÃO EXTERNA")
        print("="*60)
        
        report = classification_report(
            y_true, y_pred, 
            target_names=self.encoder.classes_,
            digits=4
        )
        print(report)
        
        # Salva relatório em arquivo
        report_path = "results/relatorio_stress_test.txt"
        with open(report_path, "w") as f:
            f.write("="*60 + "\n")
            f.write("RELATÓRIO DE VALIDAÇÃO EXTERNA (STRESS TEST)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Acurácia Geral: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write(report)
        
        print(f"\n Relatório salvo em: {report_path}")
        
        # Gera matriz de confusão
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, 
            display_labels=self.encoder.classes_, 
            cmap='OrRd',  # Cor diferente para diferenciar do treino
            xticks_rotation=45,
            ax=ax
        )
        plt.title(f"Matriz de Confusão: Validação Externa\nAcurácia: {accuracy:.2%}")
        plt.tight_layout()
        
        cm_path = "results/matriz_stress_test.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Matriz de confusão salva em: {cm_path}")
        
        # Análise de erros por classe
        print("\n Análise de Erros por Classe:")
        for i, class_name in enumerate(self.encoder.classes_):
            mask = y_true == i
            if mask.sum() > 0:
                acc_class = (y_pred[mask] == y_true[mask]).sum() / mask.sum()
                print(f"   {class_name:25s}: {acc_class:.4f} ({acc_class*100:.2f}%)")
        
        # Alertas de performance
        print("\n" + "="*60)
        print("ANÁLISE DE PERFORMANCE")
        print("="*60)
        
        if accuracy >= 0.999:
            print("ALERTA: Acurácia extremamente alta!")
            print("   Possível data leakage ou dataset muito fácil")
        elif accuracy >= 0.95:
            print(" Excelente performance!")
        elif accuracy >= 0.85:
            print(" Boa performance")
        elif accuracy >= 0.70:
            print("  Performance moderada - considere ajustes")
        else:
            print(" Performance baixa - revisar modelo e features")