import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from program2 import save_csv

def seq_feat_sel(model, folder, file, save_folder, save_file):
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    
    file_path = os.path.join(folder, file)
    df = pd.read_csv(file_path)
    
    X = df.iloc[:, 2:]
    y = df.iloc[:, 1]
    
    sfs1 = SFS(model, 
                k_features=192, 
                forward=True, 
                floating=False, 
                verbose=2,
                scoring='accuracy',
                cv=5,
                n_jobs=-1)
    
    sfs1 = sfs1.fit(X, y)
    sfs1.subsets_
    print(sfs1.k_score_)
    
    X_new = X.iloc[:, list(sfs1.k_feature_idx_)]
    df_new = pd.concat([y, X_new], axis=1)
    df_new2 = pd.concat([df.iloc[:, 0], df_new], axis=1)
    
    save_csv(save_folder, save_file, df_new2)



if __name__ == '__main__':
    knn = KNeighborsClassifier(n_neighbors=4)
    svc = SVC(kernel='rbf')
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    
    #seq_feat_sel(knn, "csv_windows_25_0", "final_window.csv", "output_files_25", "reduced_df.csv")
    seq_feat_sel(knn, "csv_windows_50_0", "final_window.csv", "output_files_50", "reduced_df.csv")