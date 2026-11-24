# Pronóstico mensual de denuncias de robo y hurto en Lima Metropolitana

**Evaluación de modelos de aprendizaje Local y Global mediante XGBoost**

Proyecto de investigación desarrollado en la **Universidad Peruana de Ciencias Aplicadas (UPC)**, como parte del curso de Capstone de Ingeniería de Sistemas.

## 1. Descripción del proyecto

Este repositorio reúne los notebooks, archivos de configuración y resultados experimentales utilizados para evaluar el pronóstico mensual de denuncias registradas de robo y hurto en los 43 distritos de Lima Metropolitana.

La investigación compara dos estrategias de entrenamiento:

- **Local:** entrenamiento independiente de un modelo XGBoost por distrito y categoría delictiva.
- **Global:** entrenamiento de un modelo XGBoost conjunto por categoría delictiva, utilizando información de los 43 distritos.

El objetivo es determinar si el entrenamiento conjunto permite reducir el error de pronóstico frente al entrenamiento independiente, utilizando condiciones experimentales comparables.

## 2. Conjunto de datos

La investigación utiliza registros del Sistema de Denuncias Policiales (SIDPOL/MININTER).

| Característica | Descripción |
|---|---|
| Fuente | SIDPOL / Ministerio del Interior del Perú |
| Período | Enero de 2018 – julio de 2026 |
| Cobertura geográfica | 43 distritos de Lima Metropolitana |
| Categorías | Robo y hurto |
| Series temporales | 86 |
| Observaciones por serie | 103 meses |
| Registros del panel | 8.858 |
| Horizonte | Un mes (h = 1) |

Durante la auditoría se aplicaron controles de continuidad, integridad, registros faltantes y valores potencialmente atípicos.

**Nota:** los valores corresponden a denuncias policiales registradas y no representan necesariamente la totalidad de delitos ocurridos.

## 3. Contenido del repositorio

Los archivos experimentales se encuentran almacenados en un archivo comprimido en formato **RAR**, que incluye los notebooks ejecutados y los resultados generados.

| Notebook | Descripción |
|---|---|
| Notebook 01 – Auditoría | Preparación de los datos, validación de integridad y construcción del panel mensual |
| Notebook 02 – Calibración | Calibración interna y selección de hiperparámetros |
| Notebook 03 – Pronósticos | Generación y verificación de pronósticos retrospectivos |
| Notebook 04 – Resultados | Cálculo de métricas, contrastes estadísticos y análisis de resultados |

Los notebooks se desarrollaron en Google Colab y sus salidas se conservaron como evidencia de las ejecuciones realizadas.

**Materiales de trabajo:** [Carpeta de Google Drive](https://drive.google.com/drive/folders/1VNsua0B07-7q1ZtV575-hRmgjGQlXsvh?usp=sharing)

## 4. Metodología experimental

El experimento comprende las siguientes etapas:

1. Auditoría y preparación de los datos SIDPOL.
2. Construcción de 86 series temporales mensuales.
3. Calibración interna utilizando los doce meses pronosticados de 2022.
4. Selección de configuraciones de XGBoost.
5. Entrenamiento Local y Global con una ventana expansiva.
6. Evaluación externa de 43 meses pronosticados, entre enero de 2023 y julio de 2026, con orígenes entre diciembre de 2022 y junio de 2026.
7. Cálculo de errores y contrastes estadísticos por distrito.

### Comparaciones experimentales

**E1 — Contraste principal**

Compara Global con Local utilizando la misma configuración de hiperparámetros H_L, obtenida mediante la calibración de los modelos locales.

**E1b — Análisis de sensibilidad**

Compara Global con su propia configuración calibrada H_G frente a Local con H_L.

Se incluyen adicionalmente los métodos de referencia Naive-1 y Naive estacional de doce meses.

## 5. Tecnologías utilizadas

Los notebooks se desarrollaron en Python mediante Google Colab.

Las versiones observadas en el entorno consultado fueron:

| Tecnología | Versión |
|---|---|
| Python | 3.13.18 |
| XGBoost | 3.4.1 |
| NumPy | 2.4.6 |
| Pandas | 2.2.3 |
| Scikit-learn | 1.6.1 |

Estas versiones corresponden al entorno consultado y deben contrastarse con los registros de las ejecuciones originales antes de utilizarlas como especificación exacta de reproducción.

## 6. Métricas de evaluación

La métrica principal fue el error absoluto medio escalado (**MASE**). Se utilizaron adicionalmente MAE y RMSE.

Las diferencias entre estrategias se analizaron mediante:

- Comparaciones pareadas de MASE por distrito.
- Intervalos de confianza del 95 % mediante bootstrap distrital.
- Pruebas de Wilcoxon con corrección de Holm.

Los pronósticos mensuales de un mismo distrito no se trataron como observaciones independientes para los contrastes estadísticos.

## 7. Resultados principales

El contraste principal E1 mostró diferencias medianas pareadas de MASE favorables a Global.

| Categoría | ΔMASE (Global − Local) | IC del 95 % |
|---|---:|---|
| Robo | −0,081 | [−0,099; −0,058] |
| Hurto | −0,066 | [−0,091; −0,046] |

En el análisis E1b, las diferencias medianas fueron −0,070 para robo y −0,069 para hurto.

No se encontró evidencia suficiente para establecer superioridad estadísticamente concluyente del enfoque Global frente al método de referencia Naive-1.

## 8. Reproducción de los experimentos

Para examinar los materiales:

1. Descargar el archivo RAR del repositorio.
2. Descomprimir los notebooks y archivos auxiliares.
3. Abrir los archivos `.ipynb` en Google Colab o un entorno compatible con Jupyter.
4. Revisar las dependencias y rutas de archivos utilizadas en cada notebook.
5. Disponer del snapshot de datos y los archivos de configuración correspondientes.
6. Ejecutar los notebooks en el orden 01 → 02 → 03 → 04, siempre que sus instrucciones y dependencias lo permitan.

Los resultados almacenados permiten inspeccionar las salidas de las ejecuciones originales. No obstante, la reproducción completa requiere disponer del conjunto de datos utilizado, las versiones originales de las dependencias y la configuración experimental.

## 9. Limitaciones

La evaluación se limita a los distritos, categorías y períodos estudiados. Además, las denuncias registradas pueden verse afectadas por cambios administrativos, diferencias de cobertura y posibles subregistros.

La evaluación retrospectiva no demuestra que los pronósticos representen la criminalidad real ni que su aplicación mejore las decisiones de seguridad ciudadana.

## 10. Autores

**Miguel Barzola Dolmos**  
**Eduardo Gabriel Santillán Taboada**

Universidad Peruana de Ciencias Aplicadas (UPC)  
Facultad de Ingeniería — Ingeniería de Sistemas

**Asesor:** Christian Antonio Cipriano Portugal

## 11. Uso académico

Este repositorio se proporciona con fines de documentación y revisión académica. La disponibilidad de los archivos no implica autorización para redistribuir datos de terceros ni garantiza la reproducción del experimento sin sus dependencias y datos de entrada originales.
