import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_prep import Prepare_data
from models.xgboost import XGBoostModel
from src.validation import ModelValidator

RAW_DATASET = "dataset/dataset.csv"
PROCESSED_DATASET = "dataset/processed_dataset.csv"

# separar uma parte do dataset para validacao
VAL_DATASET = ""

# para validacao
MODEL_RESULT = "results/xgb_mirai_model.pkl"


if __name__ == "__main__":
    prep = Prepare_data()
    
    # 1. PROCESSAMENTO INICIAL (Só precisa rodar uma vez se o arquivo não existir)
    prep.save_to_csv(RAW_DATASET, PROCESSED_DATASET)
    
    # 2. CARREGAMENTO E AMOSTRAGEM ALEATÓRIA
    # Para treinar com o dataset inteiro 'na memória', usamos uma amostra grande 
    # ou o arquivo filtrado total se a sua RAM permitir.
    print(" Carregando dados filtrados...")
    X_raw, y = prep.load_raw_data(PROCESSED_DATASET)


    #DIVISAO entre treino e teste
    print("Dividindo em 80% Treino e 20% Teste...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # O Scaler aprende a média APENAS com os dados de treino
    X_train_scaled = prep.scaler.fit_transform(X_train_raw)
    # O teste é escalado usando a regra do treino (sem ver o futuro)
    X_test_scaled = prep.scaler.transform(X_test_raw)

    # convertendo em dataframe
    X_train_final = pd.DataFrame(X_train_scaled, columns=prep.selected_features)
    X_test_final = pd.DataFrame(X_test_scaled, columns=prep.selected_features)

    # TREINAMENTO
    # Passamos o X de treino e o Scaler para que ele salve no .pkl
    print("Iniciando XGBoost...")
    XGBoostModel(X_train_final, y_train, prep.scaler)

    # VALIDAÇÃO - depois rodamos com o modelo
    # O validador agora vai carregar o Scaler do arquivo e testar no dataset original
    # validator = ModelValidator(VAL_DATASET)
    # validator.run_stress_test(VAL_DATASET, sample_size=100000)