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

# Definimos nuestras etiquetas y features
y = df['HeartDisease']
X = df.drop('HeartDisease', axis=1)

# Dividimos en sets de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# Creamos el pipeline con Random Forest Classifier
pipeline = Pipeline([
    ('rfc', RandomForestClassifier(random_state=0))
])

# Definimos los parámetros a probar en el RandomizedSearchCV
parameters = {
    'rfc__n_estimators': [50, 80, 100],
    'rfc__max_depth': [5, 8, 10],
    'rfc__min_samples_split': [2, 5, 10],
    'rfc__min_samples_leaf': [1, 2, 4],
    'rfc__class_weight': ['balanced', None]
}

# Realizamos la búsqueda aleatoria de hiperparámetros utilizando RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, parameters, cv=5, scoring='recall', random_state=0)
random_search.fit(X_train, y_train)

# Obtenemos las mejores configuraciones de hiperparámetros encontradas
best_params_RandomForest = random_search.best_params_

# Obtenemos las predicciones del mejor modelo encontrado
y_pred_RandomForest = random_search.predict(X_test)

# Obtenemos la ruta completa del directorio 'models' dentro del proyecto
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

# Guardamos las predicciones del modelo en un archivo pickle dentro de 'models'
predictions_file = os.path.join(models_dir, 'trained_model_RandomForest.pkl')
with open(predictions_file, 'wb') as file:
    pickle.dump(y_pred_RandomForest, file)








# Obtenemos la ruta completa del directorio 'data/processing' 
current_dir_test = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(current_dir_test, '..', 'data', 'test')
    
# Creamos el directorio 'data/raw' si no existe
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

 # Guardamos conjunto de prueba en archivo CSV
test_file = os.path.join(test_dir, 'test.csv')
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv(test_file, index=False)

# Obtenemos de la ruta completa del directorio 'data/processing' 
current_dir_train = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(current_dir_train, '..', 'data', 'train')
    
# Creamos el directorio 'data/train' si no existe
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

 # Guardamos el conjunto de prueba en archivo CSV
train_file = os.path.join(train_dir, 'train.csv')
train_data = pd.concat([X_train, X_test], axis=1) # DUDAS DE QUE ES LO QUE TENGO QUE CONCATENAR
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


# Obtenemos la ruta completa del directorio 'data/train' 
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(current_dir, '..', 'data', 'processed')
    
# Creamos el directorio 'data/train' si no existe
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    
# Guardamos del DataFrame como un archivo CSV en 'data/train'
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
