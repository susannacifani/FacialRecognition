import subprocess
import pandas as pd
import os
import json
from pathlib import Path
import numpy as np


in_col = ["participant_id", "behavior_1", "behavior_2", "behavior_3", "behavior_4"]
pose = ['gaze_angle_x', 'gaze_angle_y', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'p_rx', 'p_ry', 'p_rz']
au = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
pose_keypoints = ['Nose_x', 'Nose_y', 'Neck_x', 'Neck_y', 'RShoulder_x', 'RShoulder_y', 
                  'RElbow_x', 'RElbow_y', 'RWrist_x', 'RWrist_y', 'LShoulder_x', 'LShoulder_y', 
                  'LElbow_x', 'LElbow_y', 'LWrist_x', 'LWrist_y', 'MidHip_x', 'MidHip_y', 
                  'RHip_x', 'RHip_y', 'RKnee_x', 'RKnee_y', 'RAnkle_x', 'RAnkle_y', 
                  'LHip_x', 'LHip_y', 'LKnee_x', 'LKnee_y', 'LAnkle_x', 'LAnkle_y', 
                  'REye_x', 'REye_y', 'LEye_x', 'LEye_y', 'REar_x', 'REar_y', 'LEar_x', 'LEar_y', 
                  'LBigToe_x', 'LBigToe_y', 'LSmallToe_x', 'LSmallToe_y', 'LHeel_x', 'LHeel_y', 
                  'RBigToe_x', 'RBigToe_y', 'RSmallToe_x', 'RSmallToe_y', 'RHeel_x', 'RHeel_y']
hand_keypoints = ['Wrist_Lx', 'Wrist_Ly', 'Thumb1_Lx', 'Thumb1_Ly', 'Thumb2_Lx', 'Thumb2_Ly', 
                  'Thumb3_Lx', 'Thumb3_Ly', 'Thumb4_Lx', 'Thumb4_Ly', 'Index1_Lx', 'Index1_Ly', 
                  'Index2_Lx', 'Index2_Ly', 'Index3_Lx', 'Index3_Ly', 'Index4_Lx', 'Index4_Ly', 
                  'Middle1_Lx', 'Middle1_Ly', 'Middle2_Lx', 'Middle2_Ly', 'Middle3_Lx', 
                  'Middle3_Ly', 'Middle4_Lx', 'Middle4_Ly', 'Ring1_Lx', 'Ring1_Ly', 'Ring2_Lx', 
                  'Ring2_Ly', 'Ring3_Lx', 'Ring3_Ly', 'Ring4_Lx', 'Ring4_Ly', 'Pinky1_Lx', 
                  'Pinky1_Ly', 'Pinky2_Lx', 'Pinky2_Ly', 'Pinky3_Lx', 'Pinky3_Ly', 'Pinky4_Lx', 
                  'Pinky4_Ly', 'Wrist_Rx', 'Wrist_Ry', 'Thumb1_Rx', 'Thumb1_Ry', 'Thumb2_Rx', 
                  'Thumb2_Ry', 'Thumb3_Rx', 'Thumb3_Ry', 'Thumb4_Rx', 'Thumb4_Ry', 'Index1_Rx', 
                  'Index1_Ry', 'Index2_Rx', 'Index2_Ry', 'Index3_Rx', 'Index3_Ry', 'Index4_Rx', 
                  'Index4_Ry', 'Middle1_Rx', 'Middle1_Ry', 'Middle2_Rx', 'Middle2_Ry', 
                  'Middle3_Rx', 'Middle3_Ry', 'Middle4_Rx', 'Middle4_Ry', 'Ring1_Rx', 'Ring1_Ry', 
                  'Ring2_Rx', 'Ring2_Ry', 'Ring3_Rx', 'Ring3_Ry', 'Ring4_Rx', 'Ring4_Ry', 
                  'Pinky1_Rx', 'Pinky1_Ry', 'Pinky2_Rx', 'Pinky2_Ry', 'Pinky3_Rx', 'Pinky3_Ry', 
                  'Pinky4_Rx', 'Pinky4_Ry']

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
    #subprocess.run(["python", "ELANtoCSV.py"])
    
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    
    folder1 = "csv_from_elan"
    folder2 = "csv"
    folder3 = "json"
    dataframes = {}
    #lista_dataframe = []
    
    files1 = [f for f in os.listdir(folder1) if f.endswith('.csv')] #elenco dei file CSV nella cartella csv_from_elan
    for file in files1:
        file_path = os.path.join(folder1, file)
        df = pd.read_csv(file_path) #carica ciascun file CSV in un DataFrame
        dataframes[file] = df #salva il DataFrame nel dizionario con il nome del file come chiave (es 9.csv)

    files2 = [f for f in os.listdir(folder2) if f.endswith('.csv')] #elenco dei file CSV nella cartella csv
    for file in files2:
        new_filename = file[3:]
        number = new_filename.split(".")[0]
        file_path2 = os.path.join(folder2, file)
        keepCol = pose + au
        df2 = pd.read_csv(file_path2, usecols=keepCol) #carica ciascun file CSV in un DataFrame
        list_df2 = df2.values.tolist() #trasforma il DataFrame in una lista di liste
        df_col = in_col + pose + au + pose_keypoints + hand_keypoints
        bn1 = np.zeros((len(df2), len(in_col)))
        bn1[:, 0] = number
        bn2 = np.zeros((len(df2), (len(pose_keypoints) + len(hand_keypoints))))
        
        temp = []
        temprh = []
        templh = []
        num = '0' + number if len(number) == 1 else number
        for entry in Path(folder3).glob(f'**/*{num}'):
            if not entry.is_dir():
                continue
            subfolder_path = entry.resolve()
            for file_entry in subfolder_path.glob('*.json'):
                with file_entry.open() as f:
                    data = json.load(f)
                    people = data.get('people', [])
                    if not people:
                        continue
                    frame = int(file_entry.stem.split('_')[1].lstrip('0') or 0) #riga
                    el_p = people[0]['pose_keypoints_2d']
                    el_lh = people[0]['hand_left_keypoints_2d']
                    el_rh = people[0]['hand_right_keypoints_2d']
                    temp = [el_p[i:i+2] for i in range(0, len(el_p), 3)]
                    temprh = [el_rh[i:i+2] for i in range(0, len(el_rh), 3)]
                    templh = [el_lh[i:i+2] for i in range(0, len(el_lh), 3)]
                    final = temp + temprh + templh
                    #bn2
                    col = 0
                    for row in range(frame, frame+len(final)):
                        bn2[frame, col:col+len(final[row-frame])] = final[row-frame]
                        col += len(final[row-frame]) 
    
        if new_filename in dataframes:
            df1 = dataframes[new_filename] #prende il DataFrame corrispondente nel dizionario utilizzando il nome del file
            for index, row in df1.iterrows():
                start = row['inizio'] - 1
                stop = row['fine'] - 1
                action = row['classe']
                bn1[start:stop, action] = 1   

            concat_list = np.concatenate((bn1, list_df2), axis=1).tolist() #concatena i numpy array e la lista di liste
            concat_list2 = np.concatenate((concat_list, bn2), axis=1).tolist()
            
            new_df = pd.DataFrame(concat_list2, columns=(tuple(df_col)))

            new_df[in_col] = new_df[in_col].astype(int)
            new_df[pose] = new_df[pose].apply(lambda x: x * 1000).applymap('{:.0f}'.format)
            new_df[au] = new_df[au].applymap('{:.2f}'.format)
                
            #lista_dataframe.append(new_df) #aggiunge il Dataframe corrente alla lista dei Dataframe
            
            output_folder = "csv_output"
            save_csv(output_folder, new_filename, new_df) #salva il DataFrame come csv

    # name = 'final.csv'
    # concat_csv(output_folder, name, lista_dataframe) #concatena tutti i DataFrame e li salva come csv
    
    print("Programma terminato correttamente")
    
    
if __name__ == '__main__':
    csv_intersec()