import os
import pandas as pd
import ta

def preprocessing_1():
    # Ruta del archivo CSV en el directorio 'data/raw'
    csv_path = os.path.join('..', 'data', 'raw', 'GOOG.csv')

    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(csv_path)

    df = df[["Adj Close"]]  # Seleccionar la columna "Adj Close" del DataFrame original
    df.columns = ["close"]  # Cambiar el nombre de la columna a "close"
    
    def feature_engineering(df):
        # Copiamos el dataframe para evitar interferencias en los datos
        df_copy = df.copy()

        # Creamos el retorno porcentual con respecto al día de ayer
        df_copy["returns"] = df_copy["close"].pct_change(1)

        # Creamos las SMAs
        df_copy["SMA 15"] = df_copy[["close"]].rolling(15).mean().shift(1)
        df_copy["SMA 60"] = df_copy[["close"]].rolling(60).mean().shift(1)
        
        # Crear la volatilidad
        df_copy["MSD 10"] = df_copy[["returns"]].rolling(10).std().shift(1)
        df_copy["MSD 30"] = df_copy[["returns"]].rolling(30).std().shift(1)
        
        # Crear el RSI
        rsi = ta.momentum.RSIIndicator(df_copy["close"], window=14, fillna=False)
        df_copy["rsi"] = rsi.rsi()
        
        return df_copy

    # Obtener la ruta completa del directorio 'data/processed' en relación al directorio actual
    current_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

    # Crear el directorio "data/processed" si no existe
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Guardar el DataFrame procesado en el directorio "data/processed"
    processed_path = os.path.join(processed_dir, 'processed.csv')
    df_copy = feature_engineering(df)
    df_copy.to_csv(processed_path, index=False)

    return df_copy


preprocessing_1()