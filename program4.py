import os
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#start_time = time.time()

pd.options.display.max_rows = None
pd.options.display.max_columns = None

folder = "csv_windows_25_0"
file = "final_window.csv"
file_path = os.path.join(folder, file)
df = pd.read_csv(file_path)

#divide il dataframe in input X e target y
X = df.iloc[:, 2:]
y = df.iloc[:, 1]

knn = KNeighborsClassifier(n_neighbors=4)
svc = SVC(kernel='rbf')
rfc = RandomForestClassifier(n_estimators=100, random_state=0)


sfs1 = SFS(knn, 
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


folder = "output_files"
filename = "reduced_df"
df_new2.to_csv(os.path.join(folder + '/' + filename), index=False)