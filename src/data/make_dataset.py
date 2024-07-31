import pandas as pd



def load_and_preprocess_data(data_path):
    
    df = pd.read_csv(data_path)
    
    return df