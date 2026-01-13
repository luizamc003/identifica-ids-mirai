import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.data_prep import Prepare_data
from models.xgboost import XGBoostModel
from src.validation import ModelValidator

# caminhos dos arquivos
RAW_DATASET = "dataset/dataset.csv"
PROCESSED_DATASET = "dataset/processed_dataset.csv"
VAL_DATASET = "dataset/validacao_final.csv" # dataset que o modelo nunca vai ver
MODEL_RESULT = "results/xgb_mirai_model.pkl"

if __name__ == "__main__":
    # chamando classe para preparacao de dados
    prepare_data = Prepare_data()
    
    # processamento inicial - criando dataset filtrado se nao existir
    if not os.path.exists(PROCESSED_DATASET):
        print("Criando dataset processado...")
        prepare_data.save_to_csv(RAW_DATASET, PROCESSED_DATASET)
    
    # carregando os dados filtrados
    print("Carregando dados filtrados...")
    X_raw, y = prepare_data.load_raw_data(PROCESSED_DATASET)

    # primeira divisao - separando 10% para validacao externa (holdout)
    # esses dados sao salvos em um csv a parte para teste de stress
    print("Separando 10% para validacao externa em csv diferente...")
    X_rest, X_val_final, y_rest, y_val_final = train_test_split(
        X_raw, y, test_size=0.10, random_state=42, stratify=y
    )
    
    # salvando o csv de validacao
    val_df = pd.concat([X_val_final, y_val_final], axis=1)
    val_df.to_csv(VAL_DATASET, index=False)
    print(f"Dataset de validacao salvo em: {VAL_DATASET}")

    # segunda divisao - dividindo os 90% restantes em treino (80%) e teste (20%)
    print("Dividindo restante em 75% Treino e 25% Teste...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_rest, y_rest, test_size=0.25, random_state=42, stratify=y_rest
    )

    # o scaler aprende a media apenas com os dados de treino
    X_train_scaled = prepare_data.scaler.fit_transform(X_train_raw)
    
    # o teste e escalado usando a regra do treino
    X_test_scaled = prepare_data.scaler.transform(X_test_raw)

    # convertendo em dataframe para manter os nomes das colunas
    X_train_final = pd.DataFrame(X_train_scaled, columns=prepare_data.selected_features)
    X_test_final = pd.DataFrame(X_test_scaled, columns=prepare_data.selected_features)

# imprimindo colunas apos processamento e escala
    print("\nColunas do DataFrame de treino (X_train_final):")
    print(X_train_final.columns.tolist())

    # visualizando as primeiras linhas das colunas escalonadas
    print("\nVisualizacao dos dados escalonados (head):")
    print(X_train_final.head())
    
    # treinando o modelo e passando o scaler para salvar no pkl
    print("Iniciando treinamento do XGBoost...")
    # XGBoostModel(X_train_final, y_train, prepare_data.scaler)

    # validacao final usando o arquivo csv separado
    # print("Iniciando teste de stress no dataset de validacao...")
    # validator = ModelValidator(MODEL_RESULT)
    # validator.run_stress_test(VAL_DATASET, sample_size=len(val_df))