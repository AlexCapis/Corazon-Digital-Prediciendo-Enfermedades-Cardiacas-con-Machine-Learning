a id="init"></a>
<h1>Corazón Digital: Prediciendo Enfermedades Cardíacas con Machine Learning</h1>

#### Autor: [Alex Marzá Manuel](Poner link)

En el README de este proyecto se abordará la aplicación del machine learning para predecir enfermedades cardíacas, brindando una descripción detallada del trabajo realizado. En él, se proporcionará información sobre el contexto y la importancia de este problema de salud pública, destacando la necesidad de herramientas predictivas para el diagnóstico temprano.


<dl>
  <dt><a href="#introducción">1. Introducción </a></dt>
      <dd>Descripción detallada del problema y objetivo a tratar</dd>
    
  <dt><a href="#estructura">2. Estructura de carpetas</a></dt>
      <dd>Organización del proyecto</dd>
    
  <dt><a href="#split_train_test">3. </a></dt>
      <dd>Guardamos los datos de test desde el principio</dd>
    
  <dt><a href="#target">4. Target</a></dt>
      <dd>Distribución del target. ¿Desbalanceado?</dd>
    
  <dt><a href="#data_compr">5. Comprensión de variables</a></dt>
      <dd>Cómo son tus features</dd>
    
  <dt><a href="#feat_red_prelim">6. Feat. Red. Preliminar</a></dt>
      <dd>Reducción de features antes de empezar la analítica</dd>

  <dt><a href="#univariant">7. Análisis univariante</a></dt>  
      <dd>Primeras impresiones de las variables. Distribuciones</dd>
    
  <dt><a href="#bivariant">8. Análisis bivariante</a></dt>
      <dd>Búsqueda de relaciones entre las variables</dd>
  
  <dt><a href="#del_features">9. Eliminación de features</a></dt>
      <dd>Features con muchos missings o alto grado de cardinalidad</dd>
    
  <dt><a href="#duplicates">10. Duplicados</a></dt>
      <dd>Comprobamos si el DataFrame tiene duplicados</dd>
    
  <dt><a href="#missings">11. Missings</a></dt>
      <dd>Tratamos los missings</dd>
    
  <dt><a href="#errors">12. Anomalías y errores</a></dt>
      <dd>Detección de datos incoherentes</dd>
    
  <dt><a href="#outliers">13. Outliers</a></dt>
      <dd>Tratamos los outliers</dd>
    
  <dt><a href="#feat_engi">14. Feature Engineering</a></dt>
      <dd>14.1 Transformaciones</dd>
      <dd>14.2 Encodings</dd>
      <dd>14.3 Nuevas Features</dd>
      <dd>14.4 Escalados</dd>
    
  <dt><a href="#feat_reduc">15. Feature Reduction</a></dt>
      <dd>Filtrado de features por importancia</dd>
    
  <dt><a href="#choose_metric">16. Escoger métrica del modelo</a></dt>
      <dd>16.1 Métricas de clasificación</dd>
      <dd>16.2 Métricas de regresión</dd>
    
  <dt><a href="#choose_models">17. Decidir qué modelos</a></dt>
      <dd>Factores que influyen en esta decisión</dd>
    
  <dt><a href="#hyperparmeters">18. Elegir hiperparámetros</a></dt>
      <dd>Según el volumen de datos y sus tipos</dd>
    
  <dt><a href="#pipelines">19. Definimos pipelines y probamos</a></dt>
      <dd>Dependerá de cada modelo. Ejecutamos</dd>
    
  <dt><a href="#results">20. Resultados</a></dt>
      <dd>Comprobamos si el error se ajusta al problema</dd>
    
</dl>


<a id="carga_datos"></a>
<a href="#init"><p style="text-align:right;" href="#init">Volver al índice</p></a> 
# 1. íntroducción

#### Descripción del proyecto
Este proyecto se centra en la detección temprana y prevención de enfermedades cardíacas, que son una de las principales causas de muerte en los Estados Unidos. A través del análisis de factores de riesgo clave como presión arterial alta, colesterol elevado, tabaquismo, estado diabético, obesidad y falta de actividad física, se busca desarrollar un modelo de aprendizaje automático capaz de predecir la condición cardíaca de los individuos. 

Al aplicar técnicas computacionales avanzadas, se pretende identificar patrones y tendencias en los datos clínicos, lo que permitirá una detección temprana y un enfoque preventivo para mejorar la salud cardiovascular. Este proyecto busca brindar una herramienta eficiente para los profesionales de la salud en la toma de decisiones y promover una atención personalizada y preventiva para reducir la carga de las enfermedades cardíacas en la población.



#### Características del problema

<details>
<summary>Características detalladas</summary>
<p>
    
A continuación, se muestra una breve descripción con el significado de cada variable para una mejor comprensión acerca del problema a tratar.
 
**HeartDisease**: Encuestados que alguna vez informaron haber tenido una enfermedad cardíaca coronaria (CHD) o un infarto de miocardio (IM).

**IMC**: Índice de Masa Corporal (IMC).

**Smoking**: ¿Ha fumado al menos 100 cigarrillos en toda su vida? (La respuesta Sí o No).

**AlcoholDrinking**: Bebedores frecuentes (hombres adultos que toman más de 14 tragos p/semana y mujeres adultas que toman más de 7 tragos p/semana)

**Stroke**: (Alguna vez le dijeron) (usted tuvo) un accidente cerebrovascular?

**PhysicalHealth**: Su salud física, incluye enfermedades y lesiones físicas, ¿cuántos días durante los últimos 30 días su salud física no fue buena?

**MentalHealth**: Pensando en su salud mental, ¿durante cuántos días durante los últimos 30 días su salud mental no fue buena? (0-30 días).

**DiffWalking**: ¿Tiene serias dificultades para caminar o subir escaleras?

**Sex**: ¿Hombre o Mujer?

**AgeCategory**: Categoría de edad de catorce niveles.

**Race**: Valor de raza/etnicidad imputado.

**Diabetic**: (Alguna vez le dijeron) (usted tenía) diabetes?

**PhysicalActivity**: Adultos que informaron haber realizado actividad física o ejercicio durante los últimos 30 días además de su trabajo habitual.

**GenHealth**: ¿Diría usted que, en general, su salud es...?

**SleepTime**: en promedio, ¿cuántas horas duermes en un período de 24 horas?

**Asthma**: (Alguna vez le dijeron) (usted tenía) asma?

**KidneyDisease**: sin incluir cálculos renales, infección de la vejiga o incontinencia, ¿alguna vez le dijeron que tenía una enfermedad renal?

**SkinCancer**: (Alguna vez le dijeron) (usted tenía) cáncer de piel?
    
</p>
</details>

#### ¿Qué dificultades podemos encontrar?


- **Calidad y limpieza de los datos**: Los conjuntos de datos clínicos pueden contener errores y valores faltantes. Se requiere un análisis exhaustivo y técnicas de limpieza para asegurar datos de calidad.

- **Selección de características relevantes**: Con múltiples variables disponibles, es importante determinar qué características son más relevantes para predecir enfermedades cardíacas. Se necesita un análisis exploratorio y técnicas de selección de características.

- **Desequilibrio de clases**: Puede haber una proporción desigual entre casos positivos y negativos de enfermedad cardíaca. Esto puede afectar el rendimiento del modelo y requerir técnicas de muestreo o ajuste de pesos.
    <details>
    <summary>Ver imagen</summary>
    <img src="./img/encoding.png" alt="drawing" width="400"/>
    </details>
    - Usar la librería [`chardet`]().

- **Elección del modelo adecuado**: Se debe seleccionar y ajustar cuidadosamente el modelo de aprendizaje automático más adecuado para el problema. Requiere experimentación y comparación de modelos para encontrar el más efectivo.

- **Interpretación de resultados**: Comprender y comunicar los resultados del modelo de manera efectiva puede ser un desafío. Se necesita interpretar los hallazgos y explicar las predicciones de forma comprensible para diferentes audiencias.













A continuación se detallan las carpetas y los requisitos de cada una:

1. **data**: se almacenarán los datos utilizados en el proyecto. Se deben crear las siguientes subcarpetas:
   - `raw`: Contiene los datos en su formato original, sin procesar.
   - `processed`: Almacena los datos procesados después de realizar las transformaciones necesarias.
   - `train`: Contiene los datos de entrenamiento utilizados para entrenar el modelo.
   - `test`: Almacena los datos de prueba utilizados para evaluar el modelo.

2. **notebooks**: se encuentran los archivos Jupyter Notebook que contienen el desarrollo del proyecto. Se deben nombrar y numerar adecuadamente según el orden de ejecución.
   - `01_EDA.ipynb`: análisis exploratorio de datos.
   - `02_Preprocesamiento.ipynb`: transformaciones y limpiezas, incluyendo el feature engineering.
   - `03_Entrenamiento_Modelo.ipynb`: entrenamiento de modelos (mínimo 5 modelos supervisados diferentes y al menos 1 no supervisado) junto con su hiperparametrización.
   - `04_Evaluacion_Modelo.ipynb`: evaluación de los modelos (métricas de evaluación, interpretación de variables,...).
3. **src**: contiene los archivos fuente de Python que implementan las funcionalidades clave del proyecto. Los requisitos de los archivos son los siguientes:
   - `data_processing.py`: código para procesar los datos de la carpeta `data/raw` y guardar los datos procesados en la carpeta `data/processed`.
   - `model.py`: código para entrenar y guardar el modelo entrenado con el input de los datos de la carpeta `data/processed` y guardados los datasets de `data/train` y `data/test` utilizados en el entrenamiento.
   - `evaluation.py`: código para evaluar el modelo utilizando los datos de prueba de la carpeta `data/test` y generar métricas de evaluación.

4. **models**: se almacenarán los archivos relacionados con el modelo entrenado. Los requisitos son los siguientes:
   - `trained_model.pkl`: modelo entrenado guardado en formato pickle.
   - `model_config.yaml`: archivo con la configuración del modelo (parámetros)

5. **app**: contendrá los archivos necesarios para el despliegue del modelo en Streamlit u otra plataforma similar. Los requisitos son los siguientes:

   - `app.py`: código para la aplicación web que utiliza el modelo entrenado (Streamlit,...).
   - `requirements.txt`: especifica las dependencias del proyecto para poder ejecutar la aplicación.

5. **docs**: contendrá la documentación adicional relacionada con el proyecto, como las dos presentaciones u otros documentos relevantes.
    - `images`: 
    -- `presentacion.pptx`