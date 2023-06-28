

# Tratamiento de datos
import numpy as np
import pandas as pd

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns


# Modelado de datos
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ímportamos el modelo
from sklearn.ensemble import RandomForestClassifier


# Optimización del Modelo
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RandomizedSearchCV

# Para guardar los modelos
import pickle
import os

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')

# Ruta del archivo CSV en el directorio 'data/processed'
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, '..', 'data', 'processed', 'processed_heart.csv')

# Lecura del archivo CSV en un DataFrame
df = pd.read_csv(input_file)
print(len(df.columns))

# Definimos nuestras etiquetas y features
y = df['HeartDisease']
X = df.drop('HeartDisease', axis=1)

# Dividimos en sets de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Obtenemos la ruta completa del directorio 'data/processing' 
current_dir = os.getcwd()
print(current_dir)

test_dir = os.path.join(current_dir, 'PROYECTO-MACHINE-LEARNING/data/test')
print(test_dir)

# Creamos el directorio 'data/raw' si no existe
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

 # Guardamos conjunto de prueba en archivo CSV
test_file = os.path.join(test_dir, 'test.csv')
test_data = pd.concat([X_test, y_test], axis=1) # DUDAS DE QUE ES LO QUE TENGO QUE CONCATENAR
test_data.to_csv(test_file, index=False)

# Obtenemos de la ruta completa del directorio 'data/processing' 
train_dir = os.path.join(current_dir, 'PROYECTO-MACHINE-LEARNING/data/train')
    
# Creamos el directorio 'data/train' si no existe
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

 # Guardamos el conjunto de prueba en archivo CSV
train_file = os.path.join(train_dir, 'train.csv')
train_data = pd.concat([X_train, y_train], axis=1) 
train_data.to_csv(train_file, index=False)

# Obtenemos de la ruta completa del directorio 'data/test' 
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(current_dir, '..', 'data', 'processed')
    
# Creamos del directorio 'data/test' si no existe
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    
# Guardamos el DataFrame como un archivo CSV en 'data/raw'
filename = "processed_heart.csv"
filepath = os.path.join(processed_dir, filename)
df.to_csv(filepath, index=False)


# Verificamos si el archivo de prueba ya existe
if os.path.exists(test_file):
    print("El archivo de prueba ya existe. No se sobrescribirá.")
else:
    # Guardamos conjunto de prueba en archivo CSV
    test_data.to_csv(test_file, index=False)
    print("El archivo de prueba se ha guardado exitosamente.")

# Verificamos si el archivo de entrenamiento ya existe
if os.path.exists(train_file):
    print("El archivo de entrenamiento ya existe. No se sobrescribirá.")
else:
    # Guardamos conjunto de entrenamiento en archivo CSV
    train_data.to_csv(train_file, index=False)
    print("El archivo de entrenamiento se ha guardado exitosamente.")

