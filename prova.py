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
from sklearn.model_selection import RandomizedSearchCV

pd.options.display.max_rows = None
pd.options.display.max_columns = None

folder = "csvprova"
file = "final_window.csv" #10, 0: 16031 righe e 548 col
file_path = os.path.join(folder, file)
df = pd.read_csv(file_path)

#divide il dataframe in input X e target y
X = df.iloc[:, 2:]
y = df.iloc[:, 1]

# esegue la feature selection usando il modello di Random Forest con una 
# soglia mediana di selezione sul dataset di training

# divide i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# crea modello random forest con 100 alberi
rfc = RandomForestClassifier()

# seleziona le feature più importanti sul training set
sel = SelectFromModel(rfc, threshold='median')
# esegue l'addestramento del modello SelectFromModel sulla porzione di 
# training set X_train e y_train del dataset. Durante l'addestramento la funzione 
# seleziona le feature più importanti sulla base del modello di Random Forest rfc
sel.fit(X_train, y_train)

selected_feat= X_train.columns[(sel.get_support())]
#print(len(selected_feat))
print(selected_feat)

X_train_selected = sel.transform(X_train)
X_test_selected = sel.transform(X_test)

# crea un oggetto di tipo StratifiedKFold per eseguire la cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True)

# addestramento del modello Random Forest sui dati di training selezionati
rfc.fit(X_train_selected, y_train)

# cross validation del modello Random Forest sui dati di training selezionati
cv_scores = cross_val_score(rfc, X_train_selected, y_train, cv=cv)

print("Cross validation scores:", cv_scores)
print("Mean cross validation score:", cv_scores.mean())
print("Standard deviation of cross validation scores:", cv_scores.std())

# predizione dei dati di test selezionati
y_pred = rfc.predict(X_test_selected)

# calcolo dell'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# calcolo della precisione, del richiamo e dell'F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)




Output:
Index(['gaze_angle_x_min', 'gaze_angle_x_max', 'gaze_angle_x_mean',
       'gaze_angle_x_stdev', 'gaze_angle_x_skew', 'gaze_angle_x_kurt',
       'gaze_angle_y_min', 'gaze_angle_y_max', 'gaze_angle_y_mean',
       'gaze_angle_y_stdev',
       ...
       'Ring3_L_min', 'Ring3_L_mean', 'Ring3_L_stdev', 'Ring3_L_kurt',
       'Pinky1_L_min', 'Pinky1_L_mean', 'Pinky2_L_min', 'Pinky2_L_skew',
       'Pinky3_L_mean', 'Pinky4_L_min'],
      dtype='object', length=273)
Cross validation scores: [0.75857461 0.78475936 0.75044563 0.74955437 0.76604278]
Mean cross validation score: 0.7618753498564844
Standard deviation of cross validation scores: 0.01291809002726828
Accuracy: 0.7671517671517671
Precision: 0.8035909237421753
Recall: 0.6376320918777275
F1-Score: 0.6861916470229258