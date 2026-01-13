from src.data_prep import *
from src.data_analyzer import *
from models.xgboost import *
from src.validation import *

if __name__ == "__main__":
    
    # chamando classe para preparacao de dados
    prepare_data = Prepare_data()
    
    # filtrando e asrrumando o dataset
    # prepare_data.save_to_csv("dataset.csv", "teste.csv")
    
    # analisando os dados do dataset
    # analyzer = DataAnalyzer("dataset/teste.csv")
    
    # testando com um dataset pequeno
    # Lê apenas as primeiras 5000 linhas do original para ser ultra rápido
    df_mini = pd.read_csv("dataset/teste.csv", nrows=20000)

    # Salva esse dataset minúsculo para testar o código
    df_mini.to_csv("dataset/dataset_pequeno.csv", index=False)
    
    # Executa as análises
    # analyzer.run_all()
    
    # prepara os dados do csv para treinar
    X, y = prepare_data.transform("dataset/dataset_pequeno.csv") 
    
    # treinando o modelo
    XGBoostModel(X, y)
    
    # Para rodar, basta chamar a classe apontando para o arquivo de teste
    validator = ModelValidator("results/xgb_mirai_model.pkl")
    validator.run_stress_test("dataset/teste.csv")