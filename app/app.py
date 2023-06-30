import streamlit as st
import pandas as pd


# Ajustamos la pagina con un icono en el buscador y el titulo
st.set_page_config(page_title="Cardiopatías", page_icon=":heart:", layout="wide")

# Ponemos una imagen 
imagen = "../docs/imagenes/cardiopatia.jpg"
imagen_cargada = st.image(imagen)

# Ponemos un titulo a nuestra aplicación
st.title("Explorando el corazón: Utilizando datos clínicos y de estilo de vida para predecir enfermedades cardíacas")

texto = """
Introducción:

Este proyecto se centra en la detección temprana y prevención de enfermedades cardíacas, que son una 
de las principales causas de muerte en los Estados Unidos. A través del análisis de factores de riesgo clave como presión arterial alta, colesterol elevado, tabaquismo, estado diabético, obesidad y falta de actividad física, se busca desarrollar un modelo de aprendizaje automático capaz de predecir la condición cardíaca de los individuos. 
Al aplicar técnicas computacionales avanzadas, se pretende identificar patrones y tendencias en los datos clínicos, lo que permitirá una detección temprana y un enfoque preventivo para mejorar la salud cardiovascular. Este proyecto busca brindar una herramienta eficiente para los profesionales de la salud en la toma de decisiones y promover una atención personalizada y preventiva para reducir la carga de las enfermedades cardíacas en la población.
"""

st.write(texto)


menu = st.sidebar.selectbox("Selecciona la Página", ['PRINCIPAL','ORIGINAL','INGENIERIA DE DATOS', 'MODELOS', 'PRE-PRODUCCIÓN'])
# menu = st.sidebar.selectbox("Seleccionamos la página", ['Home','Filtros', 'Dataset'])

if menu == 'ORIGINAL':

    st.header('Datos originales') # Nombramos el título

     # Características
    texto2 = """
        CARACTERÍSTICAS 

    A continuación, se muestra una breve descripción con el significado de cada variable para una mejor comprensión acerca del problema a tratar.

        -HeartDisease: Encuestados que alguna vez informaron haber tenido una enfermedad cardíaca coronaria (CHD) o un infarto de miocardio (IM).
        -IMC: Índice de Masa Corporal (IMC).
        -Smoking: ¿Ha fumado al menos 100 cigarrillos en toda su vida? (La respuesta Sí o No).
        -AlcoholDrinking: Bebedores frecuentes (hombres adultos que toman más de 14 tragos p/semana y mujeres adultas que toman más de 7 tragos p/semana).
        -Stroke: ¿Alguna vez le dijeron usted tuvo un accidente cerebrovascular?
        -PhysicalHealth: Su salud física, incluye enfermedades y lesiones físicas, ¿cuántos días durante los últimos 30 días su salud física no fue buena?
        -MentalHealth: Pensando en su salud mental, ¿durante cuántos días durante los últimos 30 días su salud mental no fue buena? (0-30 días).
        -DiffWalking: ¿Tiene serias dificultades para caminar o subir escaleras?
        -Sex: ¿Hombre o Mujer?
        -AgeCategory: Categoría de edad de catorce niveles.
        -Race: Valor de raza/etnicidad imputado.
        -Diabetic: ¿Alguna vez le dijeron usted tenía diabetes?
        -PhysicalActivity: Adultos que informaron haber realizado actividad física o ejercicio durante los últimos 30 días además de su trabajo habitual.
        -GenHealth: ¿Diría usted que, en general, su salud es...?
        -SleepTime: En promedio, ¿cuántas horas duermes en un período de 24 horas?
        -Asthma: ¿Alguna vez le dijeron usted tenía asma?
        -KidneyDisease: sin incluir cálculos renales, infección de la vejiga o incontinencia, ¿alguna vez le dijeron que tenía una enfermedad renal?
        """

    st.write(texto2)



    if st.checkbox('Mostrar el DataFrame original'):
        df_procesado= pd.read_csv("../data/raw/heart_2020_cleaned.csv")
        df_procesado
    else:
        st.markdown('El dataset esta oculto')


elif menu == 'INGENIERIA DE DATOS':
    # Procesamos los datos
    st.header('Procesado de los datos: ')

    if st.checkbox('Mostrar el DataFrame procesado'):
        df_procesado= pd.read_csv("../data/processed/processed_heart.csv")
        df_procesado
    else:
        st.markdown('El dataset esta oculto')

    tab1, tab2, tab3 = st.tabs(['Heart Disease', 'Columnas Binarias','Salud'])

    with tab1:

        st.subheader('Visualización de la variable target')

        
        imagen = "../docs/imagenes/pie_plot.png"
        imagen_cargada = st.image(imagen)

    with tab2:

        st.subheader('Visualización de las variables binarias')

        
        imagen = "../docs/imagenes/binary_variables.png"
        imagen_cargada = st.image(imagen)

        texto_tab2 = '''
        #### Realización de bucle para cambiar los "No" por 0 y los "Yes" por 1 de sus respectivas columnas

        binary_columns = ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                  "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer"]


        for column in binary_columns:
        
            df[column] = df[column].map({"No": 0, "Yes": 1})

        
        #### De la columna de "Sex"

        df["Sex"][df["Sex"] == "Female"] = 0

        df["Sex"][df["Sex"] == "Male"] = 1
        
        '''
        st.write(texto_tab2)


    with tab3:

        st.subheader('Visualización de la variable Salud')

        
        imagen = "../docs/imagenes/bar_plot.png"
        imagen_cargada = st.image(imagen)

        texto_tab3 = '''
        #### Definimos los grupos de salud

        umbral_buena_salud = 5

        umbral_moderada_salud = 15

        #### Creamos una nueva columna con los grupos de salud

        df['GrupoSalud'] = np.select([df['PhysicalHealth'] < umbral_buena_salud, 
                              df['PhysicalHealth'] <= umbral_moderada_salud],
                             ['Buena salud', 'Salud moderada'], default='Mala salud')
        
        #### Cambiamos los string que le hemos puesto por valores

            category_mapping = {
                'Buena salud': 1,
                'Salud moderada': 2,
                'Mala salud': 3,
            }

            df['GrupoSalud_Ordinal'] = df['GrupoSalud'].map(category_mapping)
        '''
        st.write(texto_tab3)

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

        imagen = "../docs/imagenes/pie_plot_recall_log_regression.png"
        imagen_cargada = st.image(imagen)
        
        texto6= '''
    
        La matriz de confusión
        '''
        st.write(texto6)

        imagen = "../docs/imagenes/matriz_confusion_log_regression.png"
        imagen_cargada = st.image(imagen)


    with tab2:

        st.subheader('Descripción del modelo de Naive Bayes')


        texto7= '''
    
        La métrica de Recall 
        '''
        st.write(texto7)

        imagen = "../docs/imagenes/pie_plot_recall_naive_bayes.png"
        imagen_cargada = st.image(imagen)
        
        texto8= '''
    
        La matriz de confusión
        '''
        st.write(texto8)

        imagen = "../docs/imagenes/matriz_confusion_naive_bayes.png"
        imagen_cargada = st.image(imagen)

    with tab3:

        st.subheader('Descripción del modelo de Decision Tree Classifier')

        

        texto9= '''
    
        La métrica de Recall 
        '''
        st.write(texto9)

        imagen = "../docs/imagenes/pie_plot_recall_decision_tree.png"
        imagen_cargada = st.image(imagen)
        
        texto10= '''
    
        La matriz de confusión
        '''
        st.write(texto10)

        imagen = "../docs/imagenes/matriz_confusion_decision_tree.png"
        imagen_cargada = st.image(imagen)


    with tab4:

        st.subheader('Descripción del modelo de Random Forest Classifier')
        

        texto11= '''
    
        La métrica de Recall 
        '''
        st.write(texto11)

        imagen = "../docs/imagenes/pie_plot_recall_random_forest.png"
        imagen_cargada = st.image(imagen)
        
        texto12= '''
    
        La matriz de confusión
        '''
        st.write(texto12)

        imagen = "../docs/imagenes/matriz_confusion_Random_forest.png"
        imagen_cargada = st.image(imagen)

    with tab5:

        st.subheader('Descripción del modelo de XGB Classifier')

        texto13= '''
    
        La métrica de Recall 
        '''
        st.write(texto13)

        imagen = "../docs/imagenes/pie_plot_recall_xgb_classifier.png"
        imagen_cargada = st.image(imagen)
        
        texto14= '''
    
        La matriz de confusión
        '''
        st.write(texto14)

        imagen = "../docs/imagenes/matriz_confusion_xgb_classifier.png"
        imagen_cargada = st.image(imagen)


    with tab6:

        st.subheader('Descripción del modelo de PCA')


        texto15= '''
    
        La métrica de Recall 
        '''
        st.write(texto15)

        imagen = "../docs/imagenes/pie_plot_recall_pca.png"
        imagen_cargada = st.image(imagen)
        
        texto16= '''
    
        La matriz de confusión
        '''
        st.write(texto16)

        imagen = "../docs/imagenes/matriz_confusion_pca.png"
        imagen_cargada = st.image(imagen)



elif menu == 'PRE-PRODUCCIÓN':
    # Procesamos los datos
    st.header('Descripción de mejor modelo: ')
   
    df_comparacion= pd.read_csv("../data/comparation_models_metrics/informes_clasificacion.csv")
    df_comparacion
    
    texto4= '''
    
    En el problema que se está tratando de resolver, el objetivo principal es lograr una alta recall, ya que indica la capacidad del modelo para identificar correctamente la mayoría de los casos positivos. 

    Si nos enfocamos en el mayor recall posible, podemos observar que los modelos de RandomForest, Decision Tree y Regresión Logística muestran valores más altos de recall para la clase "1". Estos modelos tienen la capacidad de identificar correctamente la mayoría de los casos positivos.

    En contraste, los modelos de Naive Bayes y XGB Classifier presentan recalls más bajos para la clase "1", lo que indica que podrían no ser tan efectivos en la detección de casos positivos.

    Considerando todas las métricas y teniendo en cuenta tu objetivo principal de maximizar el recall, el modelo de RandomForest podría ser la mejor opción, ya que tiene unas métricas en conjunto más sólidas que el resto, aunque si bien es cierto la precisión para detectar los positivos es realmente baja en casi todos los modelos por lo que habría que profundizar más acerca de ello para poder mejorarla.
    
    '''
    st.write(texto4)