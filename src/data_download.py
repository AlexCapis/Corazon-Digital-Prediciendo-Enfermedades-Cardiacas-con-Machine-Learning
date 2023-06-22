import os
import yfinance as yf
import pandas as pd

def create_dataframe(symbol):
    data = yf.download(symbol)
    df = pd.DataFrame(data)
    
    # Obtener la ruta completa del directorio 'data/raw' en relación al directorio actual del archivo 'data_processing.py'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
    
    # Crear el directorio 'data/raw' si no existe
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    
    # Guardar el DataFrame como un archivo CSV en 'data/raw'
    filename = f"{symbol}.csv"
    filepath = os.path.join(raw_dir, filename)
    df.to_csv(filepath, index=False)
    
    return df

df= create_dataframe("GOOG")