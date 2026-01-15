import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class XGBoostModel:
    def __init__(self, X, y, scaler, label_encoder, output_dir="results"):
        """
        X: dados de treino já escalonados
        y: labels já codificados (numéricos)
        scaler: StandardScaler já treinado
        label_encoder: LabelEncoder já treinado
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Guarda o encoder e scaler que já vêm prontos
        self.encoder = label_encoder
        self.scaler = scaler
        
        # --- EXECUÇÃO ---
        print("\nIniciando treinamento do XGBoost...")
        X_test, y_test = self._train(X, y)
        self._evaluate(X_test, y_test)
        self._plot_importance(X.columns if hasattr(X, 'columns') else None)
        self._save_model()
        
        print(f"\nProcesso concluído! Resultados em: {self.output_dir}")
    
    def _train(self, X, y):
        """Divide e treina o modelo"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f" Treinando para detectar: {list(self.encoder.classes_)}")
        print(f"   Treino: {len(X_train)} amostras")
        print(f"   Teste:  {len(X_test)} amostras")
        
        self.model.fit(X_train, y_train)
        print(" Treinamento concluído!")
        
        return X_test, y_test
    
    def _evaluate(self, X_test, y_test):
        """Avalia o modelo e salva resultados"""
        print("\n Avaliando modelo...")
        predictions = self.model.predict(X_test)
        
        # Salva Relatório TXT
        report = classification_report(
            y_test, predictions, 
            target_names=self.encoder.classes_,
            digits=4
        )
        
        report_path = os.path.join(self.output_dir, "resultado_treinamento.txt")
        with open(report_path, "w") as f:
            f.write("="*60 + "\n")
            f.write("RELATÓRIO DE TREINAMENTO - XGBoost Mirai Detection\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        print(f" Relatório salvo: {report_path}")
        print(f"\n{report}")
        
        # Salva Matriz de Confusão
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, 
            display_labels=self.encoder.classes_, 
            cmap='Blues', 
            xticks_rotation=45, 
            ax=ax
        )
        plt.title("Matriz de Confusão - Conjunto de Teste")
        plt.tight_layout()
        
        cm_path = os.path.join(self.output_dir, "matriz_confusao.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n Matriz de confusão salva: {cm_path}")
    
    def _plot_importance(self, feature_names):
        """Gera gráfico das características mais importantes"""
        if feature_names is None: 
            return
        
        print("\n Gerando gráfico de importância...")
        
        plt.figure(figsize=(10, 6))
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(
            range(len(importances)), 
            [feature_names[i] for i in indices], 
            rotation=45, 
            ha='right'
        )
        plt.xlabel('Features')
        plt.ylabel('Importância')
        plt.title("Importância das Características de Rede (XGBoost)")
        plt.tight_layout()
        
        importance_path = os.path.join(self.output_dir, "importancia_features.png")
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Importância salva: {importance_path}")
    
    def _save_model(self):
        """Salva modelo, encoder e scaler juntos"""
        model_path = os.path.join(self.output_dir, "xgb_mirai_model.pkl")
        
        # Salva TUDO que será necessário para predição
        joblib.dump({
            'model': self.model, 
            'encoder': self.encoder,
            'scaler': self.scaler,
            'features': list(self.model.feature_names_in_) if hasattr(self.model, 'feature_names_in_') else None
        }, model_path)
        
        print(f"✅ Modelo completo salvo: {model_path}")