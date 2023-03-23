# FacialRecognition
program2.py come prima cosa esegue al suo interno ELANtoCSV.py
ELANtoCSV.py legge i file contenuti nella cartella elan e, servendosi dei dati contenuti in globals.py (cioè del contenuto del dizionario GROUP_IDs), li riebalora e li salva nella cartella csv_from_elan
In program2.py viene poi fatta l'intersezione di alcuni dati contenuti nelle cartelle csv_from_elan e csv (eliminando/aggiungendo colonne), i DataFrame risultanti vengono poi salvati in csv_output come file csv, inoltre vengono concatenati e salvati in un unico file csv: ciò avviene utilizzando le funzioni save_csv() e concat_csv() definite all'interno dello stesso script

program3.py chiede due parametri in input: la misura della finestra da analizzare e le righe di overlap (il numero deve essere compreso tra 0 e misura della finestra - 1)
Legge i file contenuti nella cartella csv_output e analizza ognuno in finestre, secondo i parametri forniti dall'utente: per ogni colonna vengono svolti dei calcoli (min, max, media, stdev, skew, kurt) che vengono poi inseriti come riga di un DataFrame
Il DataFrame viene poi salvato come file csv in csv_windows_{misura finestra}_{overlap} utilizzando la funzione save_csv() contenuta in program2.py, e infine vengono concatenati e salvati in un unico file csv tutti i DataFrame risultanti con la funzione concat_csv() contenuta in program2.py
