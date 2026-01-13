from src.data_prep import *
from src.data_analyzer import *
from models.xgboost import *

if __name__ == "__main__":
    
    # chamando classe para preparacao de dados
    prepare_data = Prepare_data()
    
    # filtrando e asrrumando o dataset
    # prepare_data.save_to_csv("dataset.csv", "teste.csv")
    
    # analisando os dados do dataset
    analyzer = DataAnalyzer("dataset/teste.csv")
    
    # Executa as análises
    analyzer.run_all()
    
    # prepara os dados do csv para treinar
    prep = Prepare_data()
    X, y = prep.transform("data/processed/teste.csv") 
    
    # treinando o modelo
    XGBoostModel(X, y)