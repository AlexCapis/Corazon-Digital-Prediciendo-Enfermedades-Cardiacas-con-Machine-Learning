import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle


def evaluation():
    # Directorio actual
    current_dir = os.getcwd()

    # Ruta del archivo de prueba
    test_file_path = os.path.join(current_dir, '..', 'data', 'test', 'test.csv')
    df_test = pd.read_csv(test_file_path)

    # Ruta del modelo entrenado
    model_file_path = os.path.join(current_dir, '..', 'models', 'trained_model.pkl')

    # Cargar el modelo entrenado
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)



    # Calcular el MSE en el conjunto de prueba
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")

evaluation()