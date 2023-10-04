import streamlit as st
import pandas as pd


# Ajustamos la pagina con un icono en el buscador y el titulo
st.set_page_config(page_title="Cardiopatías", page_icon=":heart:", layout="wide")

import streamlit as st

# Ponemos una imagen
imagen = "./docs/imagenes/cardiopatia.png"
imagen_cargada = st.image(imagen, use_column_width=True)

# Título de la aplicación
st.title("🚀 Descubre tu Futuro Cardíaco - Prediciendo Salud desde el Corazón ❤️")

# Introducción
st.write(
    """
    ## Introducción

    Imagina poder predecir posibles enfermedades cardíacas antes de que se manifiesten completamente. 
    ¡Eso es lo que nos proponemos en este proyecto!

    Las enfermedades cardíacas son una de las principales causas de muerte en los Estados Unidos y en todo el mundo. 
    Nuestro objetivo es utilizar datos clínicos y de estilo de vida para desarrollar un modelo de aprendizaje automático 
    que pueda predecir la condición cardíaca de las personas.

    📈 Analizaremos factores de riesgo clave, como la presión arterial alta, el colesterol elevado, el tabaquismo, la 
    diabetes, la obesidad y la falta de actividad física. Al aplicar técnicas computacionales avanzadas, identificaremos 
    patrones y tendencias en estos datos. Esto nos permitirá una detección temprana y enfoque preventivo para mejorar 
    la salud cardiovascular.

    🏥 Nuestro proyecto busca no solo proporcionar información valiosa a los profesionales de la salud para la toma de 
    decisiones, sino también promover una atención personalizada y preventiva. Aspiramos a reducir la carga de las 
    enfermedades cardíacas en nuestra población.

    ¡Acompáñanos en este viaje en la exploración del corazón y la predicción de un futuro más saludable! 💪🌟
    """
)



menu = st.sidebar.selectbox("Selecciona la Página", ['PRINCIPAL','ORIGINAL','INGENIERIA DE DATOS', 'MODELOS', 'PRE-PRODUCCIÓN'])



# Si la opción en el menú es 'ORIGINAL'
if menu == 'ORIGINAL':

    # Título y descripción de los datos originales
    st.header('📊 Datos Originales')
    st.write(
        "¡Bienvenido a la sección donde los datos se convierten en conocimiento! "
        "Aquí podrás explorar los datos que sirven como base para predecir enfermedades cardíacas. "
        "¿Estás listo para sumergirte en el mundo de los números y descubrir lo que tienen que contar?"
    )

    # Opción para mostrar u ocultar el DataFrame original
    if st.checkbox('Mostrar el DataFrame original'):
        df_procesado = pd.read_csv("./data/raw/heart_2020_cleaned.csv")
        st.write(df_procesado)
    else:
        st.markdown('El dataset está oculto. ¡Descubre sus secretos!')

    # Presentación visual de las características
    st.subheader('🔍 Características de los Datos')

    # Descripciones de las características
    st.markdown('''
        - 'HeartDisease': 'Encuestados que informaron haber tenido una enfermedad cardíaca coronaria o un infarto de miocardio.',
        - 'IMC': 'Índice de Masa Corporal (IMC).',
        - 'Smoking': '¿Ha fumado al menos 100 cigarrillos en toda su vida? (Sí/No).',
        - 'AlcoholDrinking': 'Bebedores frecuentes (hombres que toman más de 14 tragos p/semana y mujeres más de 7 tragos p/semana).',
        - 'Stroke': '¿Alguna vez le dijeron que tuvo un accidente cerebrovascular?',
        - 'PhysicalHealth': 'Número de días durante los últimos 30 días en los que su salud física no fue buena.',
        - 'MentalHealth': 'Número de días durante los últimos 30 días en los que su salud mental no fue buena (0-30 días).',
        - 'DiffWalking': '¿Tiene serias dificultades para caminar o subir escaleras?',
        - 'Sex': 'Género (Hombre/Mujer).',
        - 'AgeCategory': 'Categoría de edad en catorce niveles.',
        - 'Race': 'Valor de raza/etnicidad imputado.',
        - 'Diabetic': '¿Alguna vez le dijeron que tenía diabetes?',
        - 'PhysicalActivity': 'Realizó actividad física o ejercicio durante los últimos 30 días además de su trabajo habitual (Sí/No).',
        - 'GenHealth': 'En general, ¿diría usted que su salud es...?',
        - 'SleepTime': 'Promedio de horas de sueño en un período de 24 horas.',
        - 'Asthma': '¿Alguna vez le dijeron que tenía asma?',
        - 'KidneyDisease': '¿Alguna vez le dijeron que tenía una enfermedad renal excluyendo cálculos renales, infección de la vejiga o incontinencia?',
    '''
                
    )




elif menu == 'INGENIERIA DE DATOS':
    # Procesamos los datos
    st.header('Procesado de los datos: ')

    # Mostrar el DataFrame procesado si el usuario lo elige
    if st.checkbox('Mostrar el DataFrame procesado'):
        df_procesado = pd.read_csv("./data/processed/processed_heart.csv")
        st.write(df_procesado)
    else:
        st.markdown('El dataset está oculto.')

    # Crear pestañas para mostrar información sobre distintos aspectos de los datos
    tab1, tab2, tab3 = st.tabs(['Heart Disease', 'Columnas Binarias','Salud'])

    with tab1:
        st.subheader('Visualización de la variable target')

        # Mostrar gráfico para visualizar la variable target
        imagen = "./docs/imagenes/pie_plot.png"
        st.image(imagen, caption='Distribución de enfermedades cardíacas', use_column_width=True)

        st.write('En esta imagen, observamos cómo se representan las variables binarias en nuestros datos. Esto es crucial, ya que observamos un gran desbalanceo de los datos teniendo que en cuenta que el significado de 0 es "No" y el significado de 1 es "si.')
    with tab2:
        st.subheader('Visualización de las variables binarias')

        # Mostrar imagen de variables binarias
        imagen = "./docs/imagenes/binary_variables.png"
        st.image(imagen, caption='Variables binarias', use_column_width=True)

        st.write('En esta sección, observamos cómo se representan las variables binarias en nuestros datos. Esto es crucial para entender su impacto en la predicción de enfermedades cardíacas. Realizamos un mapeo de "No" a 0 y "Yes" a 1 para estas variables.')

    with tab3:
        st.subheader('Visualización de la variable Salud')

        # Mostrar imagen de gráfico de barras para la variable de salud
        imagen = "./docs/imagenes/bar_plot.png"
        st.image(imagen, caption='Grupos de salud', use_column_width=True)

        st.write('Aquí representamos los grupos de salud basados en la variable de salud física. Este análisis nos permite entender las diferentes categorías de salud en nuestra población de estudio. Creamos categorías como "Buena salud", "Mala Salud" y "Salud moderada" para facilitar el análisis y futuros modelos.')

        st.write('Además, convertimos estas categorías en valores numéricos para su uso en nuestros modelos. Esto nos permitirá relacionar la salud con la posibilidad de enfermedades cardíacas.')


elif menu == 'MODELOS':
    # Procesamos los datos
    st.header('Descripción de los Modelos: ')
   

    texto4= '''Tratamos distintos modelos para encontrar las mejores métricas posibles para nuestro problema como por ejemplo:

    1. Aprendizaje supervisado
        -Logistic Regression
        -Naive Bayes
        -Decision Tree Classifier
        -Random Forest Classifier
        -XGB Classifier
        

    2. Aprendizaje no supervisado:
        -PCA
    '''
    st.write(texto4)

    texto_metricas= '''
    
    Precision: Es la proporción de casos positivos identificados correctamente sobre el total de casos clasificados como positivos. En este caso, indica la capacidad del modelo para predecir correctamente los casos de enfermedad cardíaca.

    Recall: También conocido como sensibilidad o tasa de verdaderos positivos, es la proporción de casos positivos identificados correctamente sobre el total de casos reales positivos. Representa la capacidad del modelo para capturar correctamente los casos de enfermedad cardíaca.

    F1-Score: Es una medida que combina la precisión y el recall en una sola métrica. Es útil cuando se desea encontrar un equilibrio entre ambas medidas. Cuanto más cercano a 1 sea el valor del F1-Score, mejor será el desempeño del modelo.

    La matriz de confusión  es una herramienta que se utiliza para visualizar el desempeño de un modelo de clasificación. Es una tabla que muestra la cantidad de casos clasificados correctamente e incorrectamente por el modelo en cada una de las clases. Se compone de cuatro valores:

        Verdaderos positivos (True Positives, TP):
            Representa la cantidad de casos positivos que fueron clasificados correctamente por el modelo.

        Verdaderos negativos (True Negatives, TN): 
            Indica la cantidad de casos negativos que fueron clasificados correctamente por el modelo.

        Falsos positivos (False Positives, FP): 
            Muestra la cantidad de casos negativos que fueron incorrectamente clasificados como positivos por el modelo.

        Falsos negativos (False Negatives, FN): 
            Representa la cantidad de casos positivos que fueron incorrectamente clasificados como negativos por el modelo.
    '''
    st.write(texto_metricas)


    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Logistic Regression', 'Naive Bayes','Decision Tree Classifier', 'Random Forest Classifier','XGB Classifier', 'PCA'])

    with tab1:

        st.subheader('Descripción del modelo de Logistic Regression')

        texto5= '''
    
        La métrica de Recall 
        '''
        st.write(texto5)

        imagen = "./docs/imagenes/pie_plot_recall_log_regression.png"
        imagen_cargada = st.image(imagen)
        
        texto6= '''
    
        La matriz de confusión
        '''
        st.write(texto6)

        imagen = "./docs/imagenes/matriz_confusion_log_regression.png"
        imagen_cargada = st.image(imagen)


    with tab2:

        st.subheader('Descripción del modelo de Naive Bayes')


        texto7= '''
    
        La métrica de Recall 
        '''
        st.write(texto7)

        imagen = "./docs/imagenes/pie_plot_recall_naive_bayes.png"
        imagen_cargada = st.image(imagen)
        
        texto8= '''
    
        La matriz de confusión
        '''
        st.write(texto8)

        imagen = "./docs/imagenes/matriz_confusion_naive_bayes.png"
        imagen_cargada = st.image(imagen)

    with tab3:

        st.subheader('Descripción del modelo de Decision Tree Classifier')

        

        texto9= '''
    
        La métrica de Recall 
        '''
        st.write(texto9)

        imagen = "./docs/imagenes/pie_plot_recall_decision_tree.png"
        imagen_cargada = st.image(imagen)
        
        texto10= '''
    
        La matriz de confusión
        '''
        st.write(texto10)

        imagen = "./docs/imagenes/matriz_confusion_decision_tree.png"
        imagen_cargada = st.image(imagen)


    with tab4:

        st.subheader('Descripción del modelo de Random Forest Classifier')
        

        texto11= '''
    
        La métrica de Recall 
        '''
        st.write(texto11)

        imagen = "./docs/imagenes/pie_plot_recall_random_forest.png"
        imagen_cargada = st.image(imagen)
        
        texto12= '''
    
        La matriz de confusión
        '''
        st.write(texto12)

        imagen = "./docs/imagenes/matriz_confusion_Random_forest.png"
        imagen_cargada = st.image(imagen)

    with tab5:

        st.subheader('Descripción del modelo de XGB Classifier')

        texto13= '''
    
        La métrica de Recall 
        '''
        st.write(texto13)

        imagen = "./docs/imagenes/pie_plot_recall_xgb_classifier.png"
        imagen_cargada = st.image(imagen)
        
        texto14= '''
    
        La matriz de confusión
        '''
        st.write(texto14)

        imagen = "./docs/imagenes/matriz_confusion_xgb_classifier.png"
        imagen_cargada = st.image(imagen)


    with tab6:

        st.subheader('Descripción del modelo de PCA')


        texto15= '''
    
        La métrica de Recall 
        '''
        st.write(texto15)

        imagen = "./docs/imagenes/pie_plot_recall_pca.png"
        imagen_cargada = st.image(imagen)
        
        texto16= '''
    
        La matriz de confusión
        '''
        st.write(texto16)

        imagen = "./docs/imagenes/matriz_confusion_pca.png"
        imagen_cargada = st.image(imagen)



elif menu == 'PRE-PRODUCCIÓN':
    # Procesamos los datos
    st.header('Descripción de mejor modelo: ')
   
    df_comparacion= pd.read_csv("./data/comparation_models_metrics/informes_clasificacion.csv")
    df_comparacion
    
    
    texto4= '''
    
    En el problema que se está tratando de resolver, el objetivo principal es lograr una alta recall, ya que indica la capacidad del modelo para identificar correctamente la mayoría de los casos positivos. 

    Si nos enfocamos en el mayor recall posible, podemos observar que los modelos de RandomForest, Decision Tree y Regresión Logística muestran valores más altos de recall para la clase "1". Estos modelos tienen la capacidad de identificar correctamente la mayoría de los casos positivos.

    En contraste, los modelos de Naive Bayes y XGB Classifier presentan recalls más bajos para la clase "1", lo que indica que podrían no ser tan efectivos en la detección de casos positivos.

    Considerando todas las métricas y teniendo en cuenta tu objetivo principal de maximizar el recall, el modelo de RandomForest podría ser la mejor opción, ya que tiene unas métricas en conjunto más sólidas que el resto, aunque si bien es cierto la precisión para detectar los positivos es realmente baja en casi todos los modelos por lo que habría que profundizar más acerca de ello para poder mejorarla.
    
    '''
    st.write(texto4)