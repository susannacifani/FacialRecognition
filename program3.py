import pandas as pd
import numpy as np
import os
#import time

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

    #start_time = time.time()
    #leggo il file csv in un DataFrame
    df = pd.read_csv(path)
    stats = [s.split("_")[1] for s in list(map(lambda x: x.__name__, STATS))]
    df_col = []
    for c in df.columns:
        if df.columns.get_loc(c) < 5:
            df_col.append(c)
        else:
            for x in stats:
                name = c + '_' + str(x)
                df_col.append(name)
    #inizializza il DataFrame in cui scrivere i risultati
    result_df = pd.DataFrame(columns=(tuple(df_col)))
    #calcola il numero di finestre considerando la size e l'overlap
    window_count = int((len(df) - size) / (size - overlap))
    bn = np.empty((window_count, len(result_df.columns)))
    bn[:] = np.nan
    #itera sulle finestre
    for i in range(window_count):
        #seleziona le righe da considerare per la finestra corrente
        start = i * (size - overlap)
        end = start + size
        window_df = df.iloc[start:end]
        #itera sulle colonne del dataframe
        n = 5
        for ind, (col_name, col_values) in enumerate(window_df.iteritems()):
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
    result_df = pd.DataFrame(bn, columns=result_df.columns)
    #end_time = time.time()
    #print("Tempo di esecuzione:", end_time - start_time, "secondi")
    #formatta le colonne
    form = [f for f in df_col if f.startswith('participant') or f.startswith('behavior') or f.endswith('min') or f.endswith('max')]
    result_df[form] = result_df[form].applymap('{:.0f}'.format)
    return result_df





#if __name__ == "__main__":
    #window_size = 100
    #window_overlap = 9
    # accetta gli argomenti dall'utente
def main(window_size, window_overlap):

    #non mostra il conto delel righe/colonne finali
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    
    #imposta le cartelle di lavoro
    folder1 = "csv_output"
    
    lista_dataframe = []
    
    #elenco dei file CSV nella cartella csv_output
    files1 = [f for f in os.listdir(folder1) if f.endswith('.csv') and f != 'final.csv']
    #carica ciascun file CSV in un DataFrame
    for file in files1:
        file_path = os.path.join(folder1, file)
        new_df = window(file_path, window_size, window_overlap)
        #aggiunge il Dataframe corrente alla lista dei Dataframe
        lista_dataframe.append(new_df)
        
        #crea una nuova cartella se non esiste già
        output_folder = "csv_windows_" + str(window_size) + "_" + str(window_overlap)
        new_filename = file.split(".")[0] + '.csv'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        #salva il DataFrame 2 compilato in un nuovo file CSV
        new_df.to_csv(os.path.join(output_folder + '/' + new_filename), index=False)
    
    #concatena tutti i DataFrame in un unico DataFrame
    df_completo = pd.concat(lista_dataframe)
    #scrive il DataFrame combinato in un unico file CSV
    df_completo.to_csv(output_folder + '/' + 'final_window.csv', index=False)
    
    print("Programma terminato correttamente")
  
    
#check parametri
window_size = int(input("Lunghezza della finestra: "))
window_overlap = int(input("Righe di sovrapposizione: "))
    
while window_overlap < 0 or window_overlap > window_size-1:
    window_size = int(input("Lunghezza della finestra: "))
    window_overlap = int(input("Righe di sovrapposizione: "))
        
main(window_size, window_overlap)
