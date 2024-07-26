# Contexto

Cervecería y Maltería Quilmes, a través de su aplicación B2B llamada BEES, desea mejorar la experiencia de sus clientes desarrollando un nuevo módulo que recomiende productos para el próximo pedido. Para esto, se necesita un modelo de Machine Learning que prediga qué productos comprará cada cliente. Los datos necesarios incluyen transacciones y atributos de clientes. El modelo debe integrarse diariamente en la arquitectura de la app, y el equipo de producto utilizará el output del modelo para diseñar y visualizar el nuevo módulo en la aplicación.

**Datasets:**
- **transacciones.csv**: Información de compras realizadas entre mayo y agosto de 2022.
- **atributos.csv**: Información adicional sobre los clientes.

**Tareas:**
1. **EDA**: Integrar datasets, formular hipótesis, realizar visualizaciones.
2. **Data Wrangling y Modelado**: Preprocesamiento, modelado, justificación de asunciones y modelos.
3. **Evaluación**: Evaluar modelos en distintos periodos, justificar métricas y definir periodos de evaluación.
4. **Output**: Definir el output del modelo, incluyendo la cantidad y criterio de recomendaciones por cliente.

**Entregables:**
- Notebooks con los desarrollos realizados para responder los puntos de la evaluación.

# Detalles del Proceso Realizado

A continuación, se detalla todo el proceso realizado para desarrollar el modelo de Machine Learning que predice los productos que cada cliente comprará en su próximo pedido. En los notebooks se encuentran todas las explicaciones y justificaciones necesarias. Los notebooks más importantes son:

- [pipeline.ipynb]('./pipeline.ipynb'): En este notebook se realiza todo el procesamiento de datos, incluyendo el preprocesamiento, transformaciones y la ingeniería de características.

- [ejecutable.ipynb](#): En este notebook se ingresa un ID de usuario, se entrena el modelo con los datos del usuario y se devuelve la predicción de productos que es más probable que el cliente compre en su próximo pedido.

El proceso del pipeline permite escalabilidad y se puede actualizar con una frecuencia diaria, al igual que las recomendaciones.

Estos notebooks son fundamentales para entender el flujo completo del proyecto y cómo se generan las predicciones.

# [EDA](#eda)

## Carga de Datasets

Se cargaron los datasets `df_atributos` y `df_transacciones` para iniciar el análisis. Al inspeccionar ambos datasets, se observó que `df_transacciones` contiene datos de 4535 usuarios, mientras que `df_atributos` contiene datos de 4400 usuarios.

## Unificación de Datos

Se realizó un outer join entre `df_atributos` y `df_transacciones`, conservando todos los valores que no tienen coincidencia. Este paso permitió identificar usuarios presentes en uno u otro dataset pero no en ambos. Posteriormente, se eliminaron las columnas innecesarias.

## Verificación y Tratamiento de Nulos

Se identificaron valores nulos en varias columnas. Dado que no se disponía de datos adicionales para realizar imputaciones sofisticadas, se reemplazaron los valores nulos con "S/D". Este paso es crucial ya que no se contaba con datos suficientes para realizar imputaciones adecuadas, y la alternativa sería eliminar una cantidad significativa de datos.

## Verificación de Duplicados

Se identificaron y eliminaron valores duplicados para asegurar la integridad de los datos. Esta verificación se realizó comprobando registros duplicados basados en múltiples columnas clave.

## Verificación de Outliers

Se realizaron visualizaciones de boxplots e histogramas para identificar outliers en las columnas `totalVolumen`, `SkuDistintosPromediosXOrden` y `SkuDistintosToTales`. Se observó la presencia de valores atípicos que podrían estar relacionados con el volumen de compra y la cantidad de productos comprados por los clientes. Estos outliers requieren un análisis más profundo para determinar su impacto en el modelo.

## Verificación de Valores Únicos

Se ajustaron las columnas `POC` y `ACCOUNT_ID` para asegurar que los valores únicos estuvieran correctamente alineados. Se emparejaron y verificaron los valores para asegurar la consistencia de los datos. Se eliminaron las columnas redundantes para mantener el dataset limpio y ordenado.

## Cálculo de Columnas `SkuDistintosPromediosXOrden` y `SkuDistintosToTales`

Se calcularon estas columnas basándose en los datos de transacciones, asumiendo mínimas diferencias con los datos originales de atributos. Este cálculo es fundamental para tener una representación precisa de los productos distintos promedios por orden y los productos distintos totales para cada usuario.

## Transformaciones y Especificación de Tipo de Dato

Se normalizaron los nombres de las columnas según la convención PEP8 y se ajustaron los tipos de datos para asegurar la consistencia y facilitar el análisis posterior. Se derivaron características adicionales como el año, mes, día y día de la semana a partir de la columna `fecha`.

## Justificación para Incluir el Día de la Semana como Característica

Se realizó un análisis de la cantidad de pedidos según el día de la semana, observando variaciones significativas. La inclusión del día de la semana como característica en el modelo de predicción mejora la captura de patrones de compra y proporciona información contextual adicional, lo que puede llevar a recomendaciones más precisas.

## Reemplazo de Valores `volumen_total`

Se reemplazaron los valores "S/D" de `volumen_total` por -1 para facilitar el modelado posterior. Esta transformación permite tratar `volumen_total` como una variable numérica y evaluar su utilidad en el modelo.

## Reordenamiento Final del DataFrame

Se reordenó el DataFrame para agrupar las variables categóricas y asegurar una estructura coherente para el análisis y modelado posterior.

## Guardado del Dataset Limpio

Finalmente, se guardó el DataFrame limpio en la carpeta `Datasets`, asegurando que esté listo para el siguiente paso del análisis.

## Decisiones Clave

1. **Unificación de Datos mediante Outer Join**: Se eligió realizar un outer join para no perder información y poder analizar casos donde los usuarios aparecen en uno de los datasets pero no en el otro. Esto es importante para entender la cobertura y posibles faltantes en los datos.
   
2. **Reemplazo de Nulos con "S/D"**: Se decidió reemplazar los valores nulos con "S/D" debido a la falta de datos suficientes para realizar imputaciones más complejas. Esta decisión permite mantener todos los registros y evaluar cómo tratar estos valores en etapas posteriores.

3. **Eliminación de Duplicados**: La eliminación de duplicados asegura que cada registro sea único y evita el sesgo que los duplicados podrían introducir en el análisis y en el modelo.

4. **Análisis de Outliers**: Identificar outliers ayuda a comprender mejor los datos y considerar posibles transformaciones o tratamientos específicos para mejorar la calidad del modelo.

5. **Cálculo de Columnas Derivadas**: El cálculo de `SkuDistintosPromediosXOrden` y `SkuDistintosToTales` proporciona métricas adicionales que pueden ser útiles para el análisis y el modelado, mejorando la capacidad de predicción.

6. **Transformaciones y Normalización de Nombres**: La normalización de los nombres de las columnas y la especificación de tipos de datos aseguran consistencia y facilitan el trabajo con el dataset en etapas posteriores.

7. **Incluir el Día de la Semana**: Basado en el análisis de los patrones de compra, se incluyó el día de la semana como una característica para capturar mejor las variaciones en los hábitos de compra de los usuarios.

8. **Reemplazo de Valores `volumen_total`**: Reemplazar los valores "S/D" por -1 en la columna `volumen_total` permite tratarla como una variable numérica y evaluar su relevancia en el modelado.

9. **Reordenamiento del DataFrame**: Reordenar el DataFrame facilita el análisis y asegura que las variables estén agrupadas de manera lógica, mejorando la claridad y el manejo de los datos.


# [Modelado](#modelado)

## Carga de Dataset

Se cargó el dataset limpio `dataset_limpio` para continuar con el análisis y procesamiento de datos.

## Codificación de Variables Categóricas

Se utilizó `LabelEncoder` para la mayoría de las columnas categóricas, excepto para `canal`, ya que inicialmente se consideró que `canal` no reflejaba un factor de importancia. Se verificaron los valores faltantes en las columnas categóricas relevantes, y se decidió asignar un valor de 0 a los valores "S/D" en las columnas `perfil_digital`, `concentracion`, `nse` y `segmento_unico`. Se realizó una prueba inicial de codificación con `LabelEncoder` y, al observar que no se mantenía el orden implícito esperado, se optó por un mapeo específico para conservar dicho orden.

## Mapeo Específico para Columnas Categóricas

Se aplicó un mapeo específico para las columnas `perfil_digital`, `concentracion`, `nse` y `segmento_unico` para mantener un orden de importancia adecuado. Posteriormente, se eliminaron las columnas originales y se renombraron las columnas transformadas.

## Codificación de la Columna `canal`

Se utilizó One Hot Encoding para la columna `canal` para transformar las categorías en variables binarias. Esto permite que cada categoría se represente como una columna separada en el dataset.

## Análisis de Correlación

Se calculó la matriz de correlación de Pearson y Spearman para identificar las relaciones entre las variables. La matriz de correlación de Pearson mostró la relación lineal entre las diferentes variables, mientras que la matriz de correlación de Spearman capturó las relaciones no lineales. Ambas matrices se visualizaron utilizando mapas de calor para resaltar las correlaciones significativas.

### Análisis de las Matrices de Correlación

#### Correlación de Pearson

**Descripción:**
La matriz de correlación de Pearson muestra la relación lineal entre las diferentes variables del dataset. Se destacan las correlaciones significativas y las correlaciones negativas con `canal_Tradicional`, indicando que los puntos de venta tradicionales tienen características distintas a otros canales.

#### Correlación de Spearman

**Descripción:**
La matriz de correlación de Spearman mide la relación monótona entre las variables, capturando relaciones no lineales. Similar a Pearson, se destacan las correlaciones significativas y las correlaciones negativas con `canal_Tradicional`.

### Interpretación General

**Relaciones Clave:**
- Las correlaciones negativas significativas con `canal_Tradicional` sugieren que los clientes que utilizan este canal tienen comportamientos y características diferentes en comparación con otros canales.
- La falta de correlación significativa entre muchas variables indica que hay independencia entre estas, lo cual puede ser útil para identificar características únicas de diferentes segmentos.

## Feature Engineering

Se evaluaron las relaciones entre las características y la variable objetivo (`id_producto`). Para características numéricas, se utilizó la correlación de Pearson y para características categóricas, se utilizó el análisis de varianza (ANOVA). Se descartaron de la selección las variables `id_orden` y `anio` debido a su falta de relevancia.

### Evaluación de Características

**Correlación de Pearson:**
Se evaluaron las características numéricas y se seleccionaron aquellas con una correlación significativa con la variable objetivo.

**Análisis de Varianza (ANOVA):**
Se evaluaron las características categóricas y se seleccionaron aquellas con una alta variabilidad según el estadístico F.

**Resultados:**
Las características seleccionadas incluyen `perfil_digital`, `segmento_unico`, `canal_Autoservicio`, `canal_COMIDA`, `canal_Tradicional`, entre otras, basadas en la alta variabilidad observada en los resultados de ANOVA. `id_usuario` se incluyó para proporcionar recomendaciones personalizadas, aunque presenta baja variabilidad.

## Agrupación y Codificación de `id_producto`

Luego de realizar varias pruebas, se determinó que la mejor opción es agrupar por `id_orden` y codificar `id_producto` utilizando One Hot Encoding. Esto permite una representación adecuada de los productos en el dataset de entrenamiento.

## Guardado del Dataset de Entrenamiento

Se guardó el dataset de entrenamiento transformado, asegurando que esté listo para el siguiente paso del modelado.

# [Creación de Modelos](#creación-de-modelos)

## Transformación de `id_usuario` para la Predicción de Compras

### Objetivo
El objetivo es predecir las futuras compras de cada usuario basándonos en su historial de compras y características adicionales. Para ello, necesitamos decidir cómo tratar la columna `id_usuario` de manera eficiente y efectiva.

### Alternativas Evaluadas

1. **Usar `id_usuario` como un número directamente**:
   - Ventajas: Simplicidad en la implementación.
   - Desventajas: El modelo podría interpretar erróneamente los `id_usuario` como valores ordinales.

2. **One-Hot Encoding**:
   - Ventajas: Evita la interpretación ordinal y trata cada `id_usuario` como una categoría única.
   - Desventajas: Con una alta cardinalidad (2500 usuarios), resulta en una matriz muy dispersa y consume mucha memoria y computación.

3. **Label Encoding**:
   - Ventajas: Simplicidad y eficiencia en términos de memoria y computación.
   - Desventajas: Introduce una interpretación ordinal, pero en muchos modelos esto no afecta significativamente el rendimiento.

4. **Embeddings**:
   - Ventajas: Captura relaciones complejas entre usuarios y representa `id_usuario` en un espacio de menor dimensión. Es útil para modelos de redes neuronales.
   - Desventajas: Añade complejidad al modelo y no es necesario si no estamos capturando similitudes entre usuarios.

5. **No Incluir `id_usuario`**:
   - Ventajas: Simplifica el modelo y evita problemas de interpretación ordinal.
   - Desventajas: No permite capturar patrones específicos de cada usuario, lo cual es crucial para predicciones personalizadas.

## Escalado de Características

### Análisis de Datos
Las características adicionales del dataset son `perfil_digital`, `mes`, `dia`, `dia_semana`, `nse`, y `segmento_unico`. Estas características tienen diferentes rangos de valores, lo que puede causar que el modelo trate algunas características como más importantes simplemente por su escala.

### MinMaxScaler
Usar MinMaxScaler transformará los valores de todas las características a un rango común (0 a 1). Esto asegura que ninguna característica domine sobre las demás debido a su escala y permite que todas las características contribuyan de manera equitativa al aprendizaje del modelo.

### Features y Variable Objetivo
Las características seleccionadas incluyen `perfil_digital`, `mes`, `dia`, `dia_semana`, `nse`, `segmento_unico` y las variables de `canal`. La variable objetivo es la compra de productos (`id_producto`).

## División del Conjunto de Datos
Se dividió el dataset en conjuntos de entrenamiento y prueba para evaluar el rendimiento de los modelos.

## Modelos de Machine Learning para la Predicción de Compras

### Tipo de Problema
El problema que estamos abordando es la predicción de futuras compras de productos por parte de los clientes. Esto puede ser formulado como un problema de clasificación multietiqueta, donde el objetivo es predecir múltiples etiquetas (productos) simultáneamente para cada instancia (cliente).

### Modelos Seleccionados

1. **Random Forest**:
   - **Descripción**: Modelo de ensamble basado en la combinación de múltiples árboles de decisión.
   - **Ventajas**: Robusto frente al sobreajuste, capaz de manejar características tanto categóricas como numéricas, proporciona una estimación de la importancia de las características.
   - **Por qué usarlo**: Versátil y robusto, puede manejar datasets grandes y complejos.

2. **XGBoost**:
   - **Descripción**: Modelo de boosting que crea árboles de decisión secuenciales.
   - **Ventajas**: Alto rendimiento y eficiencia, maneja bien los datos desbalanceados, ofrece técnicas avanzadas de regularización.
   - **Por qué usarlo**: Conocido por su rendimiento superior en competencias de Machine Learning y por su capacidad para manejar datasets grandes y características complejas.

3. **Redes Neuronales**:
   - **Descripción**: Modelos inspirados en el cerebro humano consistentes en capas de neuronas.
   - **Ventajas**: Capacidad para modelar relaciones no lineales complejas, flexibilidad para ajustarse a diferentes tipos de datos y problemas, mejora con grandes cantidades de datos.
   - **Por qué usarlo**: Útiles para capturar patrones complejos en los datos y pueden aprovechar la estructura del dataset para hacer predicciones precisas.

### Conclusión
La combinación de estos tres modelos nos permitirá comparar diferentes enfoques y seleccionar el mejor modelo para predecir las futuras compras de los clientes. Cada modelo tiene sus propias fortalezas y puede ofrecer perspectivas únicas sobre los datos.

## Evaluación Inicial de los Modelos

### Random Forest
Se entrenó y evaluó un modelo de Random Forest utilizando el conjunto de datos de entrenamiento. Las predicciones se compararon con los valores reales para identificar coincidencias y discrepancias.

### XGBoost
Se entrenó y evaluó un modelo de XGBoost de manera similar, ajustando los hiperparámetros para optimizar el rendimiento.

### Redes Neuronales
Se diseñaron y entrenaron varias arquitecturas de redes neuronales para evaluar su capacidad de predicción, ajustando los hiperparámetros y utilizando técnicas como la normalización por lotes y la regularización para mejorar el rendimiento.

### Comparación de Resultados
Se compararon las predicciones de los tres modelos para varios usuarios seleccionados aleatoriamente. Se midieron métricas de rendimiento como la precisión, recall y F1-score para cada modelo y se evaluaron las coincidencias en las predicciones de productos comprados.

### Ajuste de Hiperparámetros
Se realizaron ajustes de hiperparámetros para los modelos de Random Forest y XGBoost, así como para las redes neuronales, utilizando técnicas de búsqueda aleatoria y grid search para identificar las configuraciones óptimas.

### Conclusión Final
La evaluación de los modelos permitió identificar las mejores configuraciones y técnicas para predecir las futuras compras de los clientes. Se guardaron los modelos entrenados y los resultados de las evaluaciones para su uso en futuras etapas del proyecto.

# [Evaluación de Modelos](#evaluación-de-modelos)

## Carga y Preparación de Datos

Se cargaron los archivos de comparaciones de modelos entrenados previamente (`comparaciones_rf.xlsx`, `comparaciones_rf_2.xlsx`, `comparaciones_xgb.xlsx`, `comparaciones_nn.xlsx`). Estos archivos contienen las predicciones de los modelos Random Forest, XGBoost y Redes Neuronales.

Se concatenaron los datos de los modelos Random Forest y se les agregó una columna indicando el tipo de modelo (`modelo`). Luego, se unieron los datos de todos los modelos en un único DataFrame y se resetearon los índices.

## Limpieza de Datos

Se identificaron y eliminaron registros que tenían más de 30 productos predichos en las Redes Neuronales, ya que estos valores no eran realistas. También se eliminaron registros con 0 productos predichos y coincidencias de predicción nulas.

## Cálculo de Métricas de Evaluación

Para evaluar los modelos se calcularon las siguientes métricas:
- **Accuracy**: Proporción de predicciones correctas sobre el total.
- **Precision**: Proporción de verdaderos positivos sobre el total de predicciones positivas.
- **Recall**: Proporción de verdaderos positivos sobre el total de verdaderos positivos y falsos negativos.
- **F1 Score**: Media armónica de la precisión y el recall.
- **Hamming Loss**: Proporción de etiquetas incorrectamente predichas.

Estas métricas se calcularon usando las siguientes fórmulas:
- **TP** (True Positives): Coincidencias en las predicciones.
- **FP** (False Positives): Productos predichos incorrectamente.
- **FN** (False Negatives): Productos reales no predichos.
- **TN** (True Negatives): Productos correctamente identificados como no comprados.

## Análisis de Métricas

### Promedio de Métricas por Modelo
Se calcularon los promedios de las métricas para cada modelo de clasificación:
- **Accuracy**: Precisión del modelo en general.
- **Precision**: Efectividad del modelo en las predicciones positivas.
- **Recall**: Capacidad del modelo para capturar los verdaderos positivos.
- **F1 Score**: Equilibrio entre precisión y recall.
- **Hamming Loss**: Proporción de errores en las predicciones.

### Mejores Valores de Métricas por Modelo
Se identificaron los mejores valores de cada métrica para cada modelo, destacando el rendimiento máximo alcanzado.

### Modelos con los Mejores Valores de Métricas, Agrupados por Modelo
Se seleccionaron los modelos con los mejores valores de métricas y se agruparon por tipo de modelo, mostrando las configuraciones y resultados óptimos.

### Top 5 Valores de Métricas por Usuario y Modelo de Clasificación
Se identificaron los cinco mejores valores de métricas para cada usuario y cada modelo, proporcionando una vista detallada del rendimiento individual.

### Modelos con Mejores Valores de Métricas por Usuario
Se analizaron los mejores valores de métricas por usuario, destacando el modelo más eficaz para cada caso específico.

### Top 5 Registros de Métricas por Modelo
Se identificaron los cinco mejores registros de métricas en general para cada modelo, mostrando los mejores resultados alcanzados.

## Análisis Final y Visualización

### Distribución de Modelos
Se creó un DataFrame para analizar la distribución de modelos que alcanzaron los mejores valores de métricas, mostrando la frecuencia y el modo de los modelos más efectivos.

### Conclusiones
Se evaluaron las configuraciones y resultados de los mejores modelos, identificando patrones y tendencias en el rendimiento de los diferentes algoritmos utilizados.

## Guardado de Resultados
Los resultados finales se guardaron en un archivo Excel (`comparaciones_final.xlsx`) para futuras referencias y análisis detallados.

# [Pruebas con modelo elegido (Redes Neuronales)](#pruebas-con-modelo-elegido-redes-neuronales)

## Carga y Preparación de Datos

Se cargaron las rutas necesarias para la ejecución del modelo y se instalaron las librerías requeridas. Posteriormente, se importaron las librerías necesarias para el desarrollo del modelo de Redes Neuronales, incluyendo TensorFlow, Keras, y otras librerías de apoyo como numpy, pandas y sklearn.

## Carga del Dataset

Se cargaron los datasets `dataset_limpio` y `dataset_entrenamiento_final`, que contienen las características de los usuarios y sus compras previas.

## Preprocesamiento de Datos

Se definió una función de preprocesamiento que ajusta las características de los datos y los escala utilizando MinMaxScaler. Además, se prepararon los datos para ser utilizados en el modelo de Redes Neuronales, dividiéndolos en conjuntos de entrenamiento y prueba.

## Definición del Modelo

Se definió un modelo de Redes Neuronales en Keras con varias capas densas y de normalización por lotes. El modelo se compiló con el optimizador Nadam y la función de pérdida `binary_crossentropy`.

## Función de Modelo

Se creó una función para entrenar y evaluar el modelo de Redes Neuronales para un usuario específico, ajustando los umbrales de predicción y generando un DataFrame con los resultados de las coincidencias entre las predicciones y los productos reales comprados por el usuario.

## Pruebas Iniciales

Se realizaron pruebas iniciales con dos usuarios específicos, evaluando el tiempo de ejecución del modelo y verificando las coincidencias en las predicciones.

## Evaluación General del Modelo

Se seleccionaron 20 usuarios al azar para evaluar el rendimiento del modelo con diferentes umbrales de predicción. Los resultados se almacenaron en un DataFrame y se guardaron en un archivo Excel para su análisis posterior.

## Cálculo de Métricas

Se calcularon las métricas de `accuracy`, `precision`, `recall`, `f1` y `hamming_loss` para evaluar el rendimiento del modelo. Estas métricas se promediaron para obtener una visión general del desempeño del modelo.

## Observaciones Generales

El modelo muestra un buen desempeño con clientes que tienen un mayor número de transacciones, obteniendo buenas métricas de precisión, recall, f1 y hamming loss. Se da prioridad a la precisión, ya que el enfoque está centrado en el usuario. Para los clientes con pocas transacciones, el modelo utiliza inferencia estadística para complementar las predicciones y mejorar la personalización.

## Ajuste de Recomendaciones

Se ajustó la cantidad de productos recomendados basándose en el promedio de productos distintos por orden de cada usuario, utilizando inferencia estadística para complementar las predicciones cuando el número de recomendaciones era menor al esperado.

## Evaluación Final

Se realizaron pruebas adicionales con usuarios seleccionados al azar, ajustando el umbral de predicción y la cantidad máxima de recomendaciones. Los resultados se guardaron en archivos Excel para su análisis final.

## Conclusiones

Luego de un exhaustivo análisis, se observan las mejores métricas con un umbral de entre 0.4 y 0.5 y un máximo de 6 productos recomendados. Este enfoque balancea las recomendaciones entre los beneficios para el usuario y para la empresa, utilizando inferencia estadística para mejorar la personalización cuando hay pocos datos disponibles.



