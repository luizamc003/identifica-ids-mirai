from src.data_prep import *
from src.data_analyzer import *

if __name__ == "__main__":
    prepare_data = Prepare_data()
    
    # testando
    # prepare_data.save_to_csv("dataset.csv", "teste.csv")
    
    # Nome do arquivo que você salvou na pasta data/processed
    analyzer = DataAnalyzer("dataset/teste.csv")
    
    # Executa as análises
    analyzer.run_all()