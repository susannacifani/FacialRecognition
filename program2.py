import subprocess
import pandas as pd
import os
import json
from pathlib import Path
import numpy as np
import math

diagonal = 735
in_col = ["participant_id", "behavior_1", "behavior_2", "behavior_3", "behavior_4"]
fin_col = ['participant_id', 'behavior']
pose = ['gaze_angle_x', 'gaze_angle_y', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'p_rx', 'p_ry', 'p_rz']
au = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

pose_keypoints = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 
                  'LShoulder', 'LElbow', 'LWrist', 
                  'MidHip', 'RHip', 'RKnee', 'RAnkle', 
                  'LHip', 'LKnee', 'LAnkle', 
                  'REye', 'LEye', 'REar', 'LEar',
                  'LBigToe', 'LSmallToe', 'LHeel', 
                  'RBigToe', 'RSmallToe', 'RHeel']

hand_keypoints = ['Wrist_L', 'Thumb1_L', 'Thumb2_L', 'Thumb3_L', 'Thumb4_L', 
                  'Index1_L', 'Index2_L', 'Index3_L', 'Index4_L',
                  'Middle1_L', 'Middle2_L', 'Middle3_L', 'Middle4_L', 
                  'Ring1_L', 'Ring2_L', 'Ring3_L', 'Ring4_L', 
                  'Pinky1_L', 'Pinky2_L', 'Pinky3_L', 'Pinky4_L', 
                  'Wrist_R', 'Thumb1_R', 'Thumb2_R', 'Thumb3_R', 'Thumb4_R', 
                  'Index1_R', 'Index2_R', 'Index3_R', 'Index4_R', 
                  'Middle1_R', 'Middle2_R', 'Middle3_R', 'Middle4_R',
                  'Ring1_R', 'Ring2_R', 'Ring3_R', 'Ring4_R',
                  'Pinky1_R', 'Pinky2_R', 'Pinky3_R', 'Pinky4_R']

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
    lista_dataframe = []
    
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
        df_col = fin_col + pose + au + pose_keypoints + hand_keypoints
        bn1 = np.zeros((len(df2), len(in_col)+1))
        bn1[:, 0] = number
        bn2 = np.zeros((len(df2), (len(pose_keypoints) + len(hand_keypoints))))
        
        temp = []
        temprh = []
        templh = []
        nose = []
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
                    nose = el_p[0:2]
                    temp = [el_p[i:i+2] for i in range(3, len(el_p), 3)]
                    temprh = [el_rh[i:i+2] for i in range(0, len(el_rh), 3)]
                    templh = [el_lh[i:i+2] for i in range(0, len(el_lh), 3)]
                    hands = temprh + templh
                    
                    dist = []
                    if nose[0] != 0 and nose[1] != 0:
                        for l in temp:
                            #if l[0] or l[1] < 1: le metto a 0 senza dist euclidea
                                #elif l[0] and l[1] != 0: distanza euclidea
                            if l[0] < 1 or l[1] < 1:
                                dist.append(0)
                            else:
                                #il naso ha coordinate (xn, yn) e un punto sulla faccia (xp, yp)
                                #d((xn,yn), (xp,yp)) = sqrt((xp - xn)^2 + (yp - yn)^2)
                                dist.append(math.sqrt((l[0] - nose[0])**2 + (l[1] - nose[1])**2))
                        for l in hands:
                            if l[0] < 1 or l[1] < 1:
                                dist.append(diagonal)
                            else:
                                dist.append(math.sqrt((l[0] - nose[0])**2 + (l[1] - nose[1])**2))
                    else:
                        dist.append(0)
                    #bn2
                    bn2[frame] = dist

    
        if new_filename in dataframes:
            df1 = dataframes[new_filename] #prende il DataFrame corrispondente nel dizionario utilizzando il nome del file
            for index, row in df1.iterrows():
                start = row['inizio'] - 1
                stop = row['fine'] - 1
                action = row['classe']
                #popolo sempre le 4 colonne di behavior con 1 
                bn1[start:stop, action] = 1
                #non serve gestire la sovrapposizione in alcun modo perchè comunque
                #eliminiamo ogni riga che ha più di un 1
                bn1[start:stop, 5] = action
                
            #elimino le righe con comportamenti sovrapposti o senza comportamenti
            index_to_delete = []
            for index, row in enumerate(bn1[:, 1:5]):
                if sum(row) > 1 or sum(row) == 0:
                    index_to_delete.append(index)
            index_to_delete.reverse() #faccio il reverse degli indici per cancellare dalla fine
                
            bn3 = np.dstack((bn1[:, 0], bn1[:, -1]))
            bn4 = bn3[0]
            concat_list = np.concatenate((bn4, list_df2), axis=1).tolist() #concatena i numpy array e la lista di liste
            concat_list2 = np.concatenate((concat_list, bn2), axis=1).tolist()
            
            for x in index_to_delete:
                del concat_list2[x]
            
            new_df = pd.DataFrame(concat_list2, columns=(tuple(df_col)))

            new_df[fin_col] = new_df[fin_col].astype(int)
            new_df[pose] = new_df[pose].apply(lambda x: x * 1000).applymap('{:.0f}'.format)
            new_df[au] = new_df[au].applymap('{:.2f}'.format)                    
                
            lista_dataframe.append(new_df) #aggiunge il Dataframe corrente alla lista dei Dataframe
            
            output_folder = "csv_output"
            save_csv(output_folder, new_filename, new_df) #salva il DataFrame come csv

    name = 'final.csv'
    concat_csv(output_folder, name, lista_dataframe) #concatena tutti i DataFrame e li salva come csv
    
    print("Programma terminato correttamente")
    
    
if __name__ == '__main__':
    csv_intersec()