import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.data_prep import Prepare_data
from models.xgboost import XGBoostModel
from src.validation import ModelValidator
from src.data_analyzer import DataAnalyzer

# Caminhos dos arquivos
RAW_DATASET = "dataset/dataset.csv"
PROCESSED_DATASET = "dataset/processed_dataset.csv"
VAL_DATASET = "dataset/validacao_final.csv"
MODEL_RESULT = "results/xgb_mirai_model.pkl"

if __name__ == "__main__":
    
    print("ANALISANDO DADOS BRUTOS:")

    # Analisa o dataset processado
    analyzer = DataAnalyzer("dataset/dataset.csv", output_dir="analise_exploratoria")
    
    # Executa todas as análises
    analyzer.run_all(features_to_plot=[
        "Header_Length", 
        "Number", 
        "Duration", 
        "Tot size",
        "IAT",  # Se ainda existir no seu dataset
        "Variance"
    ])
    
    # ============================================================
    # PREPARACAO DE DADOS
    # ============================================================
    prepare_data = Prepare_data()
    
    # Processamento inicial - criando dataset filtrado se não existir
    if not os.path.exists(PROCESSED_DATASET):
        prepare_data.save_to_csv(RAW_DATASET, PROCESSED_DATASET)
    
    # Carregando os dados filtrados
    print("\n" + "="*60)
    print("CARREGANDO DADOS")
    print("="*60)
    X_raw, y_raw = prepare_data.load_raw_data(PROCESSED_DATASET)
    print(f" Dataset carregado: {len(X_raw)} amostras, {len(X_raw.columns)} features")
    
    # ============================================================
    # CODIFICAÇÃO DOS LABELS
    # ============================================================
    print("\n" + "="*60)
    print("CODIFICANDO LABELS")
    print("="*60)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    print("Mapeamento de classes:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(y == i)
        percentage = (count / len(y)) * 100
        print(f"  {i} -> {class_name:25s} ({count:6d} amostras, {percentage:5.2f}%)")
    
    # ============================================================
    # DIAGNÓSTICO INICIAL
    # ============================================================
    print("\n" + "="*60)
    print("DIAGNÓSTICO INICIAL")
    print("="*60)
    
    # Verificando duplicatas
    duplicatas = X_raw.duplicated().sum()
    print(f"\nLinhas duplicadas: {duplicatas} ({duplicatas/len(X_raw)*100:.2f}%)")
    
    if duplicatas > 0:
        combined = pd.concat([X_raw.reset_index(drop=True), 
                             pd.Series(y, name='label')], axis=1)
        combined = combined.drop_duplicates()
        X_raw = combined[prepare_data.selected_features].reset_index(drop=True)
        y = combined['label'].values
    
    # ============================================================
    # DIVISÃO DOS DADOS
    # ============================================================
    print("\n" + "="*60)
    print("DIVISÃO DOS DADOS")
    print("="*60)
    
    # Primeira divisão - 10% para validação externa (holdout)
    print("\n Separando 10% para validação externa...")
    X_rest, X_val_final, y_rest, y_val_final = train_test_split(
        X_raw, y, test_size=0.10, random_state=42, stratify=y
    )
    
    # Salvando dataset de validação com labels originais
    val_df = pd.concat([
        X_val_final.reset_index(drop=True),
        pd.Series(label_encoder.inverse_transform(y_val_final), 
                 name='label')
    ], axis=1)
    val_df.to_csv(VAL_DATASET, index=False)
    
    # Segunda divisão - 75% treino, 25% teste dos 90% restantes
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_rest, y_rest, test_size=0.25, random_state=42, stratify=y_rest
    )
    
    print(f"    Treino: {len(X_train_raw)} amostras")
    print(f"    Teste:  {len(X_test_raw)} amostras")
    print(f"    Validação: {len(X_val_final)} amostras")
    
    # ============================================================
    # ESCALONAMENTO
    # ============================================================
    print("\n" + "="*60)
    print("ESCALONAMENTO DOS DADOS")
    print("="*60)
    
    # O scaler aprende apenas com dados de treino
    X_train_scaled = prepare_data.scaler.fit_transform(X_train_raw)
    X_test_scaled = prepare_data.scaler.transform(X_test_raw)
    
    # Convertendo para DataFrame
    X_train_final = pd.DataFrame(X_train_scaled, 
                                 columns=prepare_data.selected_features)
    X_test_final = pd.DataFrame(X_test_scaled, 
                                columns=prepare_data.selected_features)
    

    print(f"\nFeatures utilizadas ({len(prepare_data.selected_features)}):")
    for i, feat in enumerate(prepare_data.selected_features, 1):
        print(f"   {i:2d}. {feat}")
    
    # ============================================================
    # VERIFICAÇÃO DE CORRELAÇÕES
    # ============================================================
    print("\n" + "="*60)
    print("VERIFICAÇÃO DE CORRELAÇÕES")
    print("="*60)
    
    corr_matrix = X_train_final.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr = [(col, row, corr_matrix.loc[row, col]) 
                 for col in upper_triangle.columns 
                 for row in upper_triangle.index 
                 if upper_triangle.loc[row, col] > 0.95]
    
    if high_corr:
        print("\n ALERTA: Features com correlação > 0.95:")
        for feat1, feat2, corr in high_corr:
            print(f"   • {feat1:15s} x {feat2:15s}: {corr:.4f}")
        print("\nConsidere remover features redundantes no data_prep.py")
    else:
        print("Nenhuma correlação alta detectada")
    
    # Verificando variância
    variancias = X_train_final.var()
    features_sem_var = variancias[variancias < 0.01].index.tolist()
    if features_sem_var:
        print(f"\nFeatures com baixa variância: {features_sem_var}")
    
    # ============================================================
    # TREINAMENTO DO MODELO
    # ============================================================
    print("\n" + "="*60)
    print("TREINAMENTO DO MODELO")
    print("="*60)
    
    # Usando a classe XGBoostModel que você já tem
    XGBoostModel(X_train_final, y_train, prepare_data.scaler, label_encoder)
    
    # ============================================================
    # VALIDAÇÃO EXTERNA (STRESS TEST)
    # ============================================================
    print("\n" + "="*60)
    print("VALIDAÇÃO EXTERNA")
    print("="*60)
    
    print("\nIniciando teste de stress no dataset de validação...")
    validator = ModelValidator(MODEL_RESULT)
    validator.run_stress_test(VAL_DATASET)
    
    # ============================================================
    # RESUMO FINAL
    # ============================================================
    print("\n" + "="*60)
    print(" PROCESSO COMPLETO FINALIZADO!")
    print("="*60)
    print("\n Arquivos gerados:")
    print(f"   • Modelo: {MODEL_RESULT}")
    print(f"   • Validação: {VAL_DATASET}")
    print(f"   • Treino: results/resultado_treinamento.txt")
    print(f"   • Stress Test: results/relatorio_stress_test.txt")
    print(f"   • Matrizes: results/*.png")
    
    print("\n" + "="*60)