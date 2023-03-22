import subprocess
import pandas as pd
import os

subprocess.run(["python", "ELANtoCSV.py"])

#non mostra il conto delel righe/colonne finali
pd.options.display.max_rows = None
pd.options.display.max_columns = None

#imposta le cartelle di lavoro
folder1 = "csv_from_elan"
folder2 = "csv"

#crea un dizionario vuoto per contenere i DataFrame
dataframes = {}

pose = ['gaze_angle_x', 'gaze_angle_y', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'p_rx', 'p_ry', 'p_rz']
au = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']


#lista vuota per contenere i DataFrame finali
lista_dataframe = []

#elenco dei file CSV nella cartella csv_from_elan
files1 = [f for f in os.listdir(folder1) if f.endswith('.csv')]
#carica ciascun file CSV in un DataFrame
for file in files1:
    file_path = os.path.join(folder1, file)
    df = pd.read_csv(file_path)
    #salva il DataFrame nel dizionario con il nome del file come chiave (es 9.csv)
    dataframes[file] = df


#elenco dei file CSV nella cartella csv
files2 = [f for f in os.listdir(folder2) if f.endswith('.csv')]
#carica ciascun file CSV in un DataFrame
for file in files2:
    file_path2 = os.path.join(folder2, file)
    df2 = pd.read_csv(file_path2)
    #lista delle colonne da eliminare
    elimCol = list(set(df2.columns) - set(pose) - set(au))
    #rimuove le colonne che non interessano dal DataFrame, ne restano 33 con frame
    df2 = df2.drop(columns=elimCol)
    new_filename = file[3:]
    number = new_filename.split(".")[0]
    df2.insert(0, 'participant_id', number)
    df2.insert(1, 'behavior_1', '0')
    df2.insert(2, 'behavior_2', '0')
    df2.insert(3, 'behavior_3', '0')
    df2.insert(4, 'behavior_4', '0')
    
    #formatta le colonne
    for p in pose:
        df2[p] = df2[p].apply(lambda x: '{:.0f}'.format(x * 1000))
    for a in au:
        df2[a] = df2[a].map('{:.2f}'.format)
        

    if new_filename in dataframes:
        #trova il DataFrame corrispondente nel dizionario utilizzando il nome del file
        df1 = dataframes[new_filename]
        for index, row in df1.iterrows():
            start = row['inizio']
            stop = row['fine']
            action = row['classe']
            col_name = 'behavior_' + str(action)
            df2.loc[start-1:stop-1, col_name] = 1
            
            
        #aggiunge il Dataframe corrente alla lista dei Dataframe
        lista_dataframe.append(df2)
        
        #crea una nuova cartella se non esiste già
        output_folder = "csv_output"
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        #salva il DataFrame 2 compilato in un nuovo file CSV
        df2.to_csv(os.path.join(output_folder + '/' + new_filename), index=False)


#concatena tutti i DataFrame in un unico DataFrame
df_completo = pd.concat(lista_dataframe)
#scrive il DataFrame combinato in un unico file CSV
df_completo.to_csv(output_folder + '/' + 'final.csv', index=False)
