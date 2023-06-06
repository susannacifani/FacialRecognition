import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from program2 import save_csv

def normalization(folder, filename, save_file):
    file_path = os.path.join(folder, filename)
    
    data = pd.read_csv(file_path)
    features = data.iloc[:, 2:]
    scaler = StandardScaler()
    features_normalizzate = scaler.fit_transform(features)
    
    df_normalizzato = pd.DataFrame(features_normalizzate, columns=features.columns)
    df_normalizzato = pd.concat([data.iloc[:, :2], df_normalizzato], axis=1)
    
    save_csv(folder, save_file, df_normalizzato)



if __name__ == '__main__':
    # normalization("output_files_25", "reduced_df.csv", "file_normalizzato.csv")
    normalization("output_files_50", "reduced_df.csv", "file_normalizzato.csv")