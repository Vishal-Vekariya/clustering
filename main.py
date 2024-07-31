from src.data.make_dataset import load_and_preprocess_data
from src.models.train_model import Elbow_method, silhouette_method,for_visulize
from src.visualization.visualize import scattor_plot,elbow_plot, silhouette_plot
import pandas as pd


if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/mall_customers.csv"
    df = load_and_preprocess_data(data_path)
    
    wss = Elbow_method(df)
    print(wss)
    wcss = silhouette_method(df)
    print(wcss)
    dc = for_visulize(df)
    scattor_plot(dc)
    elbow_plot(wss)
    silhouette_plot(wcss)