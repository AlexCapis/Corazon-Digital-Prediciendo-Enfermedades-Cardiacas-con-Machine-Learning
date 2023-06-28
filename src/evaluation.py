
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')


# Ruta del archivo CSV en el directorio 'data/test'
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, '..', 'data', 'test', 'test.csv')

# Lecura del archivo CSV en un DataFrame
y_test = pd.read_csv(input_file)

# Seleccionamos únicamente la columna a predecir
y_test = y_test['HeartDisease']

# Cargamos el modelo desde el archivo
random_forest_ruta = os.path.join(current_dir, '..', 'models', 'trained_model_RandomForest.pkl')
mejor_modelo_RandomForest = joblib.load(random_forest_ruta)

# Generamos el informe de clasificación y lo mostramos
report = classification_report(y_test, mejor_modelo_RandomForest)
print("Classification Report:")
print(report)

# Calculamos las distintas métricas del modelo
accuracy = accuracy_score(y_test, mejor_modelo_RandomForest)
precision = precision_score(y_test, mejor_modelo_RandomForest)
recall = recall_score(y_test, mejor_modelo_RandomForest)
f1 = f1_score(y_test, mejor_modelo_RandomForest)

print("Accuracy:", accuracy)
print("Precisión:", precision)
print("Recall:", recall)
print("Score F1:", f1)

# Calculamos la matriz de confusión
cm = confusion_matrix(y_test, mejor_modelo_RandomForest)
print(cm)




