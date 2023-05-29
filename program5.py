import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# folder = "csv_windows_25_0"
# file = "ao.txt"
# file_path = os.path.join(folder, file)

# # Leggi il file di input (ad esempio, un file di testo delimitato da tabulazioni)
# #data = pd.read_csv('d_new.txt', delimiter='\t')
# data = pd.read_csv(file_path, delimiter=',')

# # Salva i dati come file CSV
# data.to_csv('output.csv', index=False)

folder = "output_files"
filename = "reduced_df.csv"
file_path = os.path.join(folder, filename)

# Leggi il file CSV
data = pd.read_csv(file_path)

# Seleziona le colonne che desideri normalizzare
features = data.iloc[:, 2:]

# Crea un oggetto StandardScaler
scaler = StandardScaler()

# Applica la normalizzazione alle colonne selezionate
features_normalizzate = scaler.fit_transform(features)

# Crea un nuovo DataFrame con le colonne normalizzate
df_normalizzato = pd.DataFrame(features_normalizzate, columns=features.columns)

# Riaggiungi le prime due colonne del DataFrame originale
df_normalizzato = pd.concat([data.iloc[:, :2], df_normalizzato], axis=1)

folder = "output_files"
filename = "file_normalizzato"
df_normalizzato.to_csv(os.path.join(folder + '/' + filename), index=False)

# Salva il DataFrame normalizzato come file CSV
#df_normalizzato.to_csv('file_normalizzato.csv', index=False)




# Calcola la media per ogni colonna
# media = data.mean()

# # Calcola la deviazione standard per ogni colonna
# deviazione_standard = data.std()

# # Stampa la media e la deviazione standard per ogni colonna
# for colonna in data.columns:
#     print(f"Colonna: {colonna}")
#     print(f"Media: {media[colonna]}")
#     print(f"Deviazione standard: {deviazione_standard[colonna]}")
#     print()