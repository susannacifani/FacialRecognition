import os
import pandas as pd
import time
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import confusion_matrix


def recall1(label4, confusion_matrix2):
    row = confusion_matrix2[label4, :]
    return confusion_matrix2[label4, label4] / row.sum()

def precision1(label5, confusion_matrix3):
    col = confusion_matrix3[:, label5]
    return confusion_matrix3[label5, label5] / col.sum()


def reports_to_txt(file_name, toScale=False, resampled=False, resampleTechnique=0):
    output_dir = r"C:\Users\susan\Documents\Python\ricfacciale\output_files"
    output_path = os.path.join(output_dir, file_name + ".txt")
    output_file = open(output_path, "w")


    folder = "output_files"
    file = "file_normalizzato.csv"
    #folder = "csv_windows_25_0"
    #file = "output.csv"
    file_path = os.path.join(folder, file)
    df = pd.read_csv(file_path)
    
    listona = ["precision", "recall", "f1-score"]
    lista = []
    tabellone = {}
    
    #divide il dataframe in input X e target y
    X = df.iloc[:, 2:]
    y = df.iloc[:, 1]
    
    
    kf = StratifiedKFold(5, shuffle=True)
    
    out = 0
    
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        
        tabella = {"precision": None, "recall": None, "f1-score": None, "accuracy": None}
        
        
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=300, stop=3000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 150, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        random_grid = {'n_estimators': n_estimators, \
                       'max_features': max_features, \
                       'max_depth': max_depth, \
                       'min_samples_split': min_samples_split, \
                       'min_samples_leaf': min_samples_leaf, \
                       'bootstrap': bootstrap}

        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=1, random_state=42, n_jobs=6)
        rf_random.fit(X_train, y_train)
        y_true, y_pred = y_test, rf_random.predict(X_test)
        

        
        print("Best parameters set found on development set:")
        print()
        print(rf_random.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        print("\nBest parameters set found on development set:\n", file=output_file)
        print(rf_random.best_params_, file=output_file)
        print("\nGrid scores on development set:\n", file=output_file)
        
        means = rf_random.cv_results_["mean_test_score"]
        stds = rf_random.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, rf_random.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=output_file)
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        
        print("\nDetailed classification report:\n", file=output_file)
        print("The model is trained on the full development set.", file=output_file)
        print("The scores are computed on the full evaluation set.\n", file=output_file)

    
        report = classification_report(y_true, y_pred, output_dict=True)
        print(report)
        print()
        
        print(report, file=output_file)
        print("\n", file=output_file)
        
        # Salvo i risultati in un file txt
        if out // 10 == 0:
            out_str = f"0{out}"
        else:
            out_str = f'{out}'

        # Confusion Matrix
        array = confusion_matrix(y_true, y_pred)
        lista.append(array)
        print(array)
        
        print(array, file=output_file)

        tabella["accuracy"] = report["accuracy"]

        print("accuracy: ", report["accuracy"])
        print("accuracy: ", report["accuracy"], file=output_file)

        for measure in listona:
            tabella[measure] = ([report[str(cls)][measure] for cls in range(1, 5)], report["macro avg"][measure],
                                report["weighted avg"][measure])

            tabellone[out_str] = tabella

    sum_array = ([[sum(matrix[i][j] for matrix in lista) for j in range(len(array[i]))] for i in range(len(array))])
    print(sum_array)
    print(sum_array, file=output_file)

    mean_of_conf_matrix_arrays = np.sum(lista, axis=0)

    print("Cofusion matrix")
    print(mean_of_conf_matrix_arrays)
    print("Cofusion matrix", file=output_file)
    print(mean_of_conf_matrix_arrays, file=output_file)

    print("Accuracy")
    print("Accuracy", file=output_file)
    diagonal_sum = mean_of_conf_matrix_arrays.trace()
    sum_of_all_elements = mean_of_conf_matrix_arrays.sum()
    print(diagonal_sum / sum_of_all_elements)
    print(diagonal_sum / sum_of_all_elements, file=output_file)

    print("label precision recall")
    print("label precision recall", file=output_file)
    # print(f"{precision('1', mean_of_conf_matrix_arrays):9.3f} {recall('1', mean_of_conf_matrix_arrays):6.3f}")

    print(recall1(0, mean_of_conf_matrix_arrays))
    print(recall1(1, mean_of_conf_matrix_arrays))
    print(recall1(2, mean_of_conf_matrix_arrays))
    print(recall1(3, mean_of_conf_matrix_arrays))
    print(precision1(0, mean_of_conf_matrix_arrays))
    print(precision1(1, mean_of_conf_matrix_arrays))
    print(precision1(2, mean_of_conf_matrix_arrays))
    print(precision1(3, mean_of_conf_matrix_arrays))

    print(recall1(0, mean_of_conf_matrix_arrays), file=output_file)
    print(recall1(1, mean_of_conf_matrix_arrays), file=output_file)
    print(recall1(2, mean_of_conf_matrix_arrays), file=output_file)
    print(recall1(3, mean_of_conf_matrix_arrays), file=output_file)
    print(precision1(0, mean_of_conf_matrix_arrays), file=output_file)
    print(precision1(1, mean_of_conf_matrix_arrays), file=output_file)
    print(precision1(2, mean_of_conf_matrix_arrays), file=output_file)
    print(precision1(3, mean_of_conf_matrix_arrays), file=output_file)

    sumf1 = 0
    sumf2 = 0
    sumf3 = 0
    sumf4 = 0
    sumpr = 0
    sumrc = 0

    print("f1")
    print("\nf1", file=output_file)
    print((2.0 * precision1(0, mean_of_conf_matrix_arrays) * recall1(0, mean_of_conf_matrix_arrays)) / (
            precision1(0, mean_of_conf_matrix_arrays) + recall1(0, mean_of_conf_matrix_arrays)))
    print((2.0 * precision1(0, mean_of_conf_matrix_arrays) * recall1(0, mean_of_conf_matrix_arrays)) / (
            precision1(0, mean_of_conf_matrix_arrays) + recall1(0, mean_of_conf_matrix_arrays)), file=output_file)
    
    sumf1 = (2.0 * precision1(0, mean_of_conf_matrix_arrays) * recall1(0, mean_of_conf_matrix_arrays)) / (
            precision1(0, mean_of_conf_matrix_arrays) + recall1(0, mean_of_conf_matrix_arrays))

    sumpr = precision1(0, mean_of_conf_matrix_arrays)
    sumrc = recall1(0, mean_of_conf_matrix_arrays)

    print((2.0 * precision1(1, mean_of_conf_matrix_arrays) * recall1(1, mean_of_conf_matrix_arrays)) / (
            precision1(1, mean_of_conf_matrix_arrays) + recall1(1, mean_of_conf_matrix_arrays)))
    print((2.0 * precision1(1, mean_of_conf_matrix_arrays) * recall1(1, mean_of_conf_matrix_arrays)) / (
            precision1(1, mean_of_conf_matrix_arrays) + recall1(1, mean_of_conf_matrix_arrays)), file=output_file)


    sumf2 = (2.0 * precision1(1, mean_of_conf_matrix_arrays) * recall1(1, mean_of_conf_matrix_arrays)) / (
            precision1(1, mean_of_conf_matrix_arrays) + recall1(1, mean_of_conf_matrix_arrays))

    sumpr += precision1(1, mean_of_conf_matrix_arrays)
    sumrc += recall1(1, mean_of_conf_matrix_arrays)

    print((2.0 * precision1(2, mean_of_conf_matrix_arrays) * recall1(2, mean_of_conf_matrix_arrays)) / (
            precision1(2, mean_of_conf_matrix_arrays) + recall1(2, mean_of_conf_matrix_arrays)))
    print((2.0 * precision1(2, mean_of_conf_matrix_arrays) * recall1(2, mean_of_conf_matrix_arrays)) / (
            precision1(2, mean_of_conf_matrix_arrays) + recall1(2, mean_of_conf_matrix_arrays)), file=output_file)



    sumf3 = (2.0 * precision1(2, mean_of_conf_matrix_arrays) * recall1(2, mean_of_conf_matrix_arrays)) / (
            precision1(2, mean_of_conf_matrix_arrays) + recall1(2, mean_of_conf_matrix_arrays))

    sumpr += precision1(2, mean_of_conf_matrix_arrays)
    sumrc += recall1(2, mean_of_conf_matrix_arrays)

    print((2.0 * precision1(3, mean_of_conf_matrix_arrays) * recall1(3, mean_of_conf_matrix_arrays)) / (
            precision1(3, mean_of_conf_matrix_arrays) + recall1(3, mean_of_conf_matrix_arrays)))
    print((2.0 * precision1(3, mean_of_conf_matrix_arrays) * recall1(3, mean_of_conf_matrix_arrays)) / (
            precision1(3, mean_of_conf_matrix_arrays) + recall1(3, mean_of_conf_matrix_arrays)), file=output_file)



    sumf4 = (2.0 * precision1(3, mean_of_conf_matrix_arrays) * recall1(3, mean_of_conf_matrix_arrays)) / (
            precision1(3, mean_of_conf_matrix_arrays) + recall1(3, mean_of_conf_matrix_arrays))

    sumpr += precision1(3, mean_of_conf_matrix_arrays)
    sumrc += recall1(3, mean_of_conf_matrix_arrays)

    print("macro Fscore")
    print((sumf1 + sumf2 + sumf3 + sumf4) / 4)

    print("macro Precision")
    print(sumpr / 4)

    print("macro Recall")
    print(sumrc / 4)
    
    
    print("\nmacro Fscore", file=output_file)
    print((sumf1 + sumf2 + sumf3 + sumf4) / 4, file=output_file)
    print("macro Precision", file=output_file)
    print(sumpr / 4, file=output_file)
    print("macro Recall", file=output_file)
    print(sumrc / 4, file=output_file)

    row1 = 0
    row2 = 0
    row3 = 0
    row4 = 0

    row1 = mean_of_conf_matrix_arrays[0, :]
    row2 = mean_of_conf_matrix_arrays[1, :]
    row3 = mean_of_conf_matrix_arrays[2, :]
    row4 = mean_of_conf_matrix_arrays[3, :]

    print("weighted Fscore")
    print((sumf1 * row1.sum() + sumf2 * row2.sum() + sumf3 * row3.sum() + sumf4 * row4.sum()) / (
                row1.sum() + row2.sum() + row3.sum() + row4.sum()))

    print("\nweighted Fscore", file=output_file)
    print((sumf1 * row1.sum() + sumf2 * row2.sum() + sumf3 * row3.sum() + sumf4 * row4.sum()) / (
                row1.sum() + row2.sum() + row3.sum() + row4.sum()), file=output_file)
    output_file.close()  
        

reports_to_txt("25fps_rf", toScale=True, resampled=False)




