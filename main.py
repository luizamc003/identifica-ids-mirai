from data_prep import *

if __name__ == "__main__":
    prepare_data = Prepare_data()
    
    # testando
    prepare_data.save_to_csv("dataset.csv", "teste.csv")