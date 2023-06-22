import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

def train_linear_regression_model():
    # Directorio actual
    current_dir = os.getcwd()

    # Directorio del archivo procesado
    file_path = os.path.join(current_dir, '..', 'data', 'processed', 'processed.csv')
    df = pd.read_csv(file_path)

    # Eliminar filas con valores faltantes en las características y la variable objetivo
    df.dropna(subset=['SMA 15', 'SMA 60', 'MSD 10', 'MSD 30', 'rsi', 'returns'], inplace=True)

    # Columnas de características
    features = df[['SMA 15', 'SMA 60', 'MSD 10', 'MSD 30', 'rsi']]

    # Variable objetivo
    target = df['returns']

    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Directorios de salida
    train_dir = os.path.join(current_dir, '..', 'data', 'train')
    test_dir = os.path.join(current_dir, '..', 'data', 'test')

    # Crear directorios de salida si no existen
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Guardar conjunto de entrenamiento en archivo CSV
    train_file = os.path.join(train_dir, 'train.csv')
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.to_csv(train_file, index=False)

    # Guardar conjunto de prueba en archivo CSV
    test_file = os.path.join(test_dir, 'test.csv')
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv(test_file, index=False)

    # Resto del código para la regresión lineal
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Predicciones en el conjunto de prueba
    y_pred = reg.predict(X_test)

    # Calcular el MSE en el conjunto de prueba
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")

    # Directorio del archivo de modelos
    models_dir = os.path.join(current_dir, '..', 'models')

    # Crear el directorio "models" si no existe
    os.makedirs(models_dir, exist_ok=True)

    # Guardar las predicciones en formato pickle
    predictions_path = os.path.join(models_dir, 'trained_model.pkl')
    with open(predictions_path, 'wb') as file:
        pickle.dump(y_pred, file)

train_linear_regression_model()