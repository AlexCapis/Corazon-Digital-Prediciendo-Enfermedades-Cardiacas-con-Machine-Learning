# PROYECTO-MACHINE-LEARNING
<a id="init"></a>
<h1>Machine Learning Guide</h1>

#### Autor: [Daniel Ortiz López](https://www.linkedin.com/in/daniel-ortiz-l%C3%B3pez/)

En este notebook encontrarás una guía de cómo afrontar un problema de Machine Learning supervisado de clasificación o regresión. Está compuesto de explicaciones teóricas, ayudas en la toma de decisiones, código y enlaces de apoyo.

<dl>
  <dt><a href="#carga_datos">1. Carga datos</a></dt>
      <dd>Formato de los datos y cantidad de archivos</dd>
    
  <dt><a href="#reg_clas">2. Problema Machine Learning</a></dt>
      <dd>Clasificación o Regresión</dd>
    
  <dt><a href="#split_train_test">3. Divide en train y test</a></dt>
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