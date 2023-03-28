import pandas as pd
import numpy as np
import os
from program2 import save_csv, concat_csv

def calc_min(col_values):
    return col_values.min()

def calc_max(col_values):
    return col_values.max()

def calc_mean(col_values):
    return col_values.mean()

def calc_stdev(col_values):
    return col_values.std(ddof=1)

def calc_skew(col_values):
    return col_values.skew()

def calc_kurt(col_values):
    return col_values.kurtosis()


STATS = [calc_min, calc_max, calc_mean, calc_stdev, calc_skew, calc_kurt]


def window(path, size, overlap):
    """
    
    Analizza un file csv in finestre, effettua calcoli sui dati delle finestre e li scrive in un nuovo DataFrame.

    Parameters
    ----------
    path : str
        Percorso in cui si trova il Dataframe da leggere.
    size : int
        Lunghezza finestra.
    overlap : int 
        Dimensione sovrapposizione finestre (in righe).

    Returns
    -------
    result_df : pandas.DataFrame
        Il DataFrame contenente i calcoli effettuati sui dati estrapolati.

    """
    df = pd.read_csv(path) #legge il file csv in un DataFrame
    stats = [s.split("_")[1] for s in list(map(lambda x: x.__name__, STATS))]
    df_col = []
    for c in df.columns:
        if df.columns.get_loc(c) < 5:
            df_col.append(c)
        else:
            for x in stats:
                name = c + '_' + str(x)
                df_col.append(name)
    result_df = pd.DataFrame(columns=(tuple(df_col))) #inizializza il DataFrame in cui scrivere i risultati
    window_count = int((len(df) - size) / (size - overlap)) #calcola il numero di finestre considerando la size e l'overlap
    bn = np.empty((window_count, len(result_df.columns)))
    bn[:] = np.nan
    for i in range(window_count): #itera sulle finestre
        start = i * (size - overlap)
        end = start + size
        window_df = df.iloc[start:end]
        n = 5
        for ind, (col_name, col_values) in enumerate(window_df.iteritems()): #itera sulle colonne del dataframe
            if 0 < ind < 5:                
                if col_values.sum() > size / 2:
                    bn[i, ind] = '1'
                else:
                    bn[i, ind] = '0'
            elif ind > 4:
                for stat_func in STATS:
                    bn[i, n] = stat_func(col_values)
                    n += 1
            else:
                bn[i, ind] = col_values.iloc[1]
    result_df = pd.DataFrame(bn, columns=result_df.columns) #scrive sul nuovo DataFrame la riga appena elaborata 
    for f in df_col:
        if f.startswith('participant') or f.startswith('behavior'):
            result_df[f] = result_df[f].apply('{:.0f}'.format)
        elif f.endswith('min') or f.endswith('max'):
            if f.startswith('AU'):
                result_df[f] = result_df[f].apply('{:.2f}'.format)
            else:
                result_df[f] = result_df[f].apply('{:.0f}'.format)
        elif (f.endswith('mean') and f.startswith('AU')) or f.endswith('stdev') or f.endswith('skew') or f.endswith('kurt'):
            result_df[f] = result_df[f].apply('{:.6f}'.format)
    return result_df


def main(window_size, window_overlap):
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    
    folder1 = "csv_output"
    lista_dataframe = []
    
    files1 = [f for f in os.listdir(folder1) if f.endswith('.csv') and f != 'final.csv']
    for file in files1:
        file_path = os.path.join(folder1, file)
        new_df = window(file_path, window_size, window_overlap) #analizza il DataFrame corrente in finestre, new_df contiene il DataFrame con nuovi dati
        lista_dataframe.append(new_df)
        output_folder = "csv_windows_" + str(window_size) + "_" + str(window_overlap)
        new_filename = file.split(".")[0] + '.csv'
        save_csv(output_folder, new_filename, new_df)

    name = 'final_window.csv'
    concat_csv(output_folder, name, lista_dataframe)
    
    print("Programma terminato correttamente")
  


if __name__ == '__main__':
    window_size = int(input("Lunghezza della finestra: "))
    window_overlap = int(input("Righe di sovrapposizione: "))
       
    while window_overlap < 0 or window_overlap > window_size-1: #check parametri inseriti
        window_overlap = int(input("Righe di sovrapposizione: "))
    main(window_size, window_overlap)