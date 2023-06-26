import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Ruta del archivo CSV en el directorio 'data/raw'
# csv_path = os.path.join('..', 'data', 'raw', 'heart_2020_cleaned.csv') 
current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.abspath(os.path.join(current_dir, '..'))
input_file = os.path.join(current_dir, '..', 'data', 'raw', 'heart_2020_cleaned.csv')


# Lecura del archivo CSV en un DataFrame
df = pd.read_csv(input_file)

# Realización de bucle para cambiar los "No" por 0 y los "Yes" por 1 de sus respectivas columnas
binary_columns = ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                  "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer"]

for column in binary_columns:
    df[column] = df[column].map({"No": 0, "Yes": 1})

# Realización de un bucle para convertir las columnas que he cambiado a binarias pues transformarlas de su formato object a int.
columns_to_convert = ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                        "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer"]

for column in columns_to_convert:
    df[column] = df[column].astype(int)

# De la columna de "Sex"
df["Sex"][df["Sex"] == "Female"] = 0
df["Sex"][df["Sex"] == "Male"] = 1

# Cambio de tipo objet a tipo int
df["Sex"] = df["Sex"].astype(int)

# Columnas a las que se les aplicará el LabelEncoder
columns = ['AgeCategory', 'Race', 'Diabetic', 'GenHealth']

# Creamos una instancia de LabelEncoder
label_encoder = LabelEncoder()

# Realización de un bucle y aplicación del LabelEncoder
for col in columns:
    df[col + '_encoded'] = label_encoder.fit_transform(df[col])

# La variable "BMI", la cual nos muestra el ídice de masa corporal
# Tengo que redondear los valores pra que no se creen valores nulos
df["BMI"] = df["BMI"].round() 

# Definición de las categorías y los límites de los rangos de IMC
categories = {
    'Bajo peso': (0, 18.4),
    'Peso normal': (18.5, 24.9),
    'Sobrepeso': (25, 29.9),
    'Obesidad clase I': (30, 34.9),
    'Obesidad clase II': (35, 39.9),
    'Obesidad clase III (obesidad mórbida)': (40, float('inf'))
}

# Realización de una función para aplicar un bucle sobre la variable más fácilmente
def assign_category(bmi):
    for category, (lower, upper) in categories.items():
        if lower <= bmi <= upper:
            return category

# Aplicamos la funciñon a la variable y creamos una nueva columna
df['BMI_Category'] = df['BMI'].apply(assign_category)

# Cambiamos los string que le hemos puesto por valores
category_mapping = {
    'Bajo peso': 1,
    'Peso normal': 2,
    'Sobrepeso': 3,
    'Obesidad clase I': 4,
    'Obesidad clase II': 5,
    'Obesidad clase III (obesidad mórbida)': 6
}
df['BMI_Category_Ordinal'] = df['BMI_Category'].map(category_mapping)

# Definimos los grupos de salud
umbral_buena_salud = 5
umbral_moderada_salud = 15

# Creamos una nueva columna con los grupos de salud
df['GrupoSalud'] = np.select([df['PhysicalHealth'] < umbral_buena_salud, 
                              df['PhysicalHealth'] <= umbral_moderada_salud],
                             ['Buena salud', 'Salud moderada'], default='Mala salud')

# Cambiamos los string que le hemos puesto por valores
category_mapping = {
    'Buena salud': 1,
    'Salud moderada': 2,
    'Mala salud': 3,
}
df['GrupoSalud_Ordinal'] = df['GrupoSalud'].map(category_mapping)

# Definimos los grupos de salud mental
umbral_buena_salud_mental = 5
umbral_moderada_salud_mental = 15

# Creamos una nueva columna con los grupos de salud
df['GrupoSalud_Mental'] = np.select([df['MentalHealth'] < umbral_buena_salud_mental, 
                              df['MentalHealth'] <= umbral_moderada_salud_mental],
                             ['Buena salud', 'Salud moderada'], default='Mala salud')

# Cambiamos los string que le hemos puesto por valores
category_mapping_mental = {
    'Buena salud': 1,
    'Salud moderada': 2,
    'Mala salud': 3,
}
df['GrupoSalud_Mental_Ordinal'] = df['GrupoSalud_Mental'].map(category_mapping_mental)

# Definimos los grupos de "SleepTime"
insuficiente_limit = 6
optimo_lower_limit = 7
optimo_upper_limit = 9

# Crear una nueva columna 'SleepGroup' basada en los grupos específicos
bins = [0, insuficiente_limit, optimo_lower_limit, float('inf')]
labels = ['Insuficiente', 'Óptimo', 'Excesivo']
df['SleepGroup'] = pd.cut(df['SleepTime'], bins=bins, labels=labels, right=False)

# Imprimir la cuenta de valores en cada grupo
df['SleepGroup'].value_counts()

# Cambiamos los string que le hemos puesto por valores
category_mapping_sleep = {
    'Insuficiente': 1,
    'Óptimo': 2,
    'Excesivo': 3,
}
df['SleepGroup_Ordinal'] = df['SleepGroup'].map(category_mapping_sleep)

# Eliminamos las columnas que son de tipo object y algunas de tipo float
df = df.drop(columns=(["AgeCategory", "SleepTime", "SleepGroup", "GenHealth","BMI_Category","GrupoSalud","GrupoSalud_Mental", "Race", "Diabetic", "PhysicalHealth", "BMI","MentalHealth"]))

# Obtención de la ruta completa del directorio 'data/processing' 
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(current_dir, '..', 'data', 'processed')
    
# Creación del directorio 'data/raw' si no existe
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    
# Guardado del DataFrame como un archivo CSV en 'data/raw'
filename = "processed_heart.csv"
filepath = os.path.join(processed_dir, filename)
df.to_csv(filepath, index=False)
