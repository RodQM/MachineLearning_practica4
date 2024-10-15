import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Paso 1: Cargar los datasets
# Utilizaremos datasets incluidos en scikit-learn para simplificar
breast_cancer = load_breast_cancer()
digits = load_digits()

# Convertir los datasets a DataFrames de Pandas para facilitar el manejo
data_breast_cancer = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
data_breast_cancer['target'] = breast_cancer.target
data_digits = pd.DataFrame(data=digits.data, columns=digits.feature_names)
data_digits['target'] = digits.target

# Paso 2: Separar las características (X) y las etiquetas (y)
X1, y1 = data_breast_cancer.iloc[:, :-1], data_breast_cancer.iloc[:, -1]
X2, y2 = data_digits.iloc[:, :-1], data_digits.iloc[:, -1]

# Paso 3: Implementar los métodos de validación

# 1. Hold Out
def hold_out_validation(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nHold Out Validation")
    print(f"Tamaño de entrenamiento: {X_train.shape}, Tamaño de prueba: {X_test.shape}")
    print(f"Precisión del modelo: {accuracy:.2f}")
    return X_train, X_test, y_train, y_test

# 2. K-Fold Cross-Validation
def k_fold_validation(X, y, k):
    kf = KFold(n_splits=k)
    accuracies = []
    print("\nK-Fold Cross Validation")
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = LogisticRegression(max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Fold {i+1}: Tamaño de entrenamiento: {X_train.shape}, Tamaño de prueba: {X_test.shape}, Precisión: {accuracy:.2f}")
    print(f"Precisión promedio: {sum(accuracies)/len(accuracies):.2f}")

# 3. Leave-One-Out
def leave_one_out_validation(X, y):
    loo = LeaveOneOut()
    accuracies = []
    print("\nLeave-One-Out Validation")
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = LogisticRegression(max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Iteración {i+1}: Tamaño de entrenamiento: {X_train.shape}, Tamaño de prueba: {X_test.shape}, Precisión: {accuracy:.2f}")
        # Limitar el número de impresiones para mantenerlo legible
        if i >= 4:
            print("...")
            break
    print(f"Precisión promedio: {sum(accuracies)/len(accuracies):.2f}")

# Paso 4: Preguntar al usuario qué método aplicar y con qué parámetros
if __name__ == "__main__":
    print("Selecciona un dataset para usar:")
    print("1: Breast Cancer")
    print("2: Digits")
    dataset_choice = int(input("Ingresa el número del dataset que deseas usar: "))
    
    if dataset_choice == 1:
        X, y = X1, y1
    elif dataset_choice == 2:
        X, y = X2, y2
    else:
        print("Opción no válida. Saliendo del programa.")
        exit()
    
    print("\nSelecciona el método de validación:")
    print("1: Hold Out")
    print("2: K-Fold Cross Validation")
    print("3: Leave-One-Out")
    method_choice = int(input("Ingresa el número del método que deseas aplicar: "))
    
    if method_choice == 1:
        test_size = float(input("Ingresa el valor de r (proporción de prueba, entre 0 y 1): "))
        hold_out_validation(X, y, test_size)
    elif method_choice == 2:
        k = int(input("Ingresa el número de folds (K): "))
        k_fold_validation(X, y, k)
    elif method_choice == 3:
        leave_one_out_validation(X, y)
    else:
        print("Opción no válida. Saliendo del programa.")