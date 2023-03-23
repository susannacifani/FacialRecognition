import subprocess
import pandas as pd
import os


def save_csv(folder, filename, df):
    """
    
    Salva un DataFrame come file CSV nella cartella specificata (se non esiste, la crea).

    Parameters
    ----------
    folder : str
        Il percorso della cartella in cui salvare il file CSV.
    filename : str
        Il nome con cui salvare il DataFrame come file CSV.
    df : pandas.DataFrame
        Il DataFrame da salvare come file CSV.

    Returns
    -------
    None.

    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    df.to_csv(os.path.join(folder + '/' + filename), index=False)
    
def concat_csv(folder, filename, df_list):
    """
    
    Concatena tutti i DataFrame contenuti nella lista e li salva in un file CSV.

    Parameters
    ----------
    folder : str
        Il percorso della cartella in cui salvare il file CSV.
    filename : str
        Il nome con cui salvare il DataFrame come file CSV.
    df_list : list
        Lista di DataFrame da concatenare in un unico DataFrame.

    Returns
    -------
    None.

    """
    df_completo = pd.concat(df_list)
    df_completo.to_csv(folder + '/' + filename, index=False)
    

def csv_intersec():
    subprocess.run(["python", "ELANtoCSV.py"])
    
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    
    folder1 = "csv_from_elan"
    folder2 = "csv"
    dataframes = {}
    pose = ['gaze_angle_x', 'gaze_angle_y', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'p_rx', 'p_ry', 'p_rz']
    au = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
    lista_dataframe = []
    
    
    files1 = [f for f in os.listdir(folder1) if f.endswith('.csv')] #elenco dei file CSV nella cartella csv_from_elan
    for file in files1:
        file_path = os.path.join(folder1, file)
        df = pd.read_csv(file_path) #carica ciascun file CSV in un DataFrame
        dataframes[file] = df #salva il DataFrame nel dizionario con il nome del file come chiave (es 9.csv)
    
    files2 = [f for f in os.listdir(folder2) if f.endswith('.csv')] #elenco dei file CSV nella cartella csv
    for file in files2:
        file_path2 = os.path.join(folder2, file)
        df2 = pd.read_csv(file_path2) #carica ciascun file CSV in un DataFrame
        elimCol = list(set(df2.columns) - set(pose) - set(au)) #lista delle colonne da eliminare
        df2 = df2.drop(columns=elimCol)
        new_filename = file[3:]
        number = new_filename.split(".")[0]
        df2.insert(0, 'participant_id', number)
        df2.insert(1, 'behavior_1', '0')
        df2.insert(2, 'behavior_2', '0')
        df2.insert(3, 'behavior_3', '0')
        df2.insert(4, 'behavior_4', '0')
        
        df2[pose] = df2[pose].apply(lambda x: x * 1000).applymap('{:.0f}'.format)
        df2[au] = df2[au].applymap('{:.2f}'.format)
    
        if new_filename in dataframes:
            df1 = dataframes[new_filename] #prende il DataFrame corrispondente nel dizionario utilizzando il nome del file
            for index, row in df1.iterrows():
                start = row['inizio']
                stop = row['fine']
                action = row['classe']
                col_name = 'behavior_' + str(action)
                df2.loc[start-1:stop-1, col_name] = 1
                
            lista_dataframe.append(df2) #aggiunge il Dataframe corrente alla lista dei Dataframe
            
            output_folder = "csv_output"
            save_csv(output_folder, new_filename, df2) #salva il DataFrame come csv

    name = 'final.csv'
    concat_csv(output_folder, name, lista_dataframe) #concatena tutti i DataFrame e li salva come csv
    
    print("Programma terminato correttamente")
    
    
if __name__ == '__main__':
    csv_intersec()