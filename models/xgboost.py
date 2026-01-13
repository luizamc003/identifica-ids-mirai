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
    #  Adicione o parâmetro 'scaler' no init
    def __init__(self, X, y, scaler, output_dir="results"): 
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42
        )
        self.encoder = LabelEncoder()

        # --- EXECUÇÃO ---
        X_test, y_test = self._train(X, y)
        self._evaluate(X_test, y_test)
        self._plot_importance(X.columns if hasattr(X, 'columns') else None)
        
        # PASSE O SCALER AQUI
        self._save_model(scaler) 
        print(f"\n Processo concluído! Resultados em: {self.output_dir}")

    def _train(self, X, y):
        y_encoded = self.encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Treinando XGBoost para detectar: {list(self.encoder.classes_)}")
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def _evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        
        # Salva Relatório TXT
        report = classification_report(y_test, predictions, target_names=self.encoder.classes_)
        with open(os.path.join(self.output_dir, "resultado_treinamento.txt"), "w") as f:
            f.write(report)

        # Salva Matriz de Confusão
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, display_labels=self.encoder.classes_, 
            cmap='Blues', xticks_rotation=45, ax=ax
        )
        plt.title("Matriz de Confusão - XGBoost Mirai Detection")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "matriz_confusao.png"))
        plt.close()

    def _plot_importance(self, feature_names):
        """Gera gráfico das características mais importantes para a segurança"""
        if feature_names is None: return

        plt.figure(figsize=(10, 6))
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title("Importância das Características de Rede (XGBoost)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "importancia_features.png"))
        plt.close()

    def _save_model(self, scaler): # Receba o scaler aqui
        model_path = os.path.join(self.output_dir, "xgb_mirai_model.pkl")
        joblib.dump({
            'model': self.model, 
            'encoder': self.encoder,
            'scaler': scaler  # SALVE O SCALER AQUI
        }, model_path)