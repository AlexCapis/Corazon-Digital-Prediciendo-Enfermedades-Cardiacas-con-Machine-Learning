import os
import yfinance as yf
import pandas as pd

def create_dataframe(symbol):
    data = yf.download(symbol)
    df = pd.DataFrame(data)
    
    # Obtención de la ruta completa del directorio 'data/raw' en relación al directorio actual del archivo 'data_processing.py'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(current_dir, '..', 'data', 'raw')
    
    # Creación del directorio 'data/raw' si no existe
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    
    # Guardado del DataFrame como un archivo CSV en 'data/raw'
    filename = f"{symbol}.csv"
    filepath = os.path.join(raw_dir, filename)
    df.to_csv(filepath, index=False)
    
    return df

 # Preguntar al usuario que acción desea visualizar,
 # le ofreceríamos una lista, ya que se necesita una nomenclatura especial mediante acrónimos
df= create_dataframe("GOOG")