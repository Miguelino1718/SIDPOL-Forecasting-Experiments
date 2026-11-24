# SIDPOL: pronóstico Local vs Global de denuncias de robo y hurto en Lima

## La pregunta en una frase

¿Entrenar un modelo con los 43 distritos (**Global**) permite pronosticar mejor el mes siguiente que entrenar un modelo por distrito (**Local**)? Robo y Hurto se estudian por separado. El panel auditado comprende enero de 2018–julio de 2026; la evaluación temporal externa abarca enero de 2023–julio de 2026.

El aporte es **comparativo y empírico**: documentar cómo cambia el desempeño al agrupar el entrenamiento, con qué configuración y frente a qué referencias. La variable estudiada son **denuncias registradas**, no la criminalidad real.

## Las cuatro notebooks

| Notebook | Qué hace en la ruta entregada | ¿Entrena en modo `revisar`? |
|---|---|---|
| **01 Auditoría** | Documenta la construcción del panel: 43 distritos, 103 meses y 2 targets; 86 series y 8,858 filas. La auditoría está cerrada y sus salidas se reutilizan. | No se ejecuta en esta ruta. |
| **02 Calibración** | Reconstruye los scores de 2022, comprueba los hiperparámetros congelados y genera ejemplos y cuatro gráficos de calibración. | No. |
| **03 Pronósticos** | Verifica los pronósticos registrados, muestra los cortes temporales y genera dos gráficos de series reales. | No. |
| **04 Resultados** | Recalcula métricas y comparaciones E1–E10 desde esos pronósticos; exporta las tablas del paper y diez figuras. | No; esta notebook siempre analiza pronósticos guardados. |

Orden de ejecución: **02 → 03 → 04**. Las notebooks comprueban huellas de archivos, correspondencia entre configuración y scores, cobertura de los pronósticos y cortes temporales. También registran el entorno utilizado. Los controles específicos se muestran junto a cada cálculo.

## Cómo ejecutar el paquete

### Revisar los resultados incluidos

1. Descomprime **toda** la carpeta `SIDPOL_Notebooks`, conservando sus subcarpetas. Trabaja sobre una copia si deseas conservar también las salidas visuales originales.
2. Abre las notebooks desde esa carpeta. En Colab, ajusta `PROJECT_DIR` en la primera celda de cada una: debe apuntar a la carpeta que contiene `sidpol.py`. En Jupyter local, abre el entorno desde esa misma carpeta.
3. Mantén esta configuración en 02, 03 y 04:

   ```python
   MODO = 'revisar'
   CARPETA_RESULTADOS = 'resultados'
   ```

4. Ejecuta todas las celdas de 02, después las de 03 y finalmente las de 04. Si reemplazaste los módulos Python, reinicia el kernel o la sesión antes de volver a ejecutar.
5. Comprueba los mensajes finales `READY_FOR_NOTEBOOK_03`, `READY_FOR_NOTEBOOK_04` y `Análisis finalizado`, respectivamente.

En esta ruta, **02 conserva la configuración y las predicciones internas; 03 conserva los pronósticos externos**. Ambas renuevan sus ejemplos, figuras e índices dentro de `figuras_notebook/`. La 04 valida primero las entradas y después renueva `04_resultados/`. Si falta un archivo o una huella no coincide, restaura los archivos de la misma entrega; ejecutar `revisar` no reconstruye una entrada dañada.

### Dependencias y versiones

`requirements_modelado.txt` contiene las versiones que el código exige para **iniciar un nuevo entrenamiento**. Para instalarlas en el entorno de Python que utilizarás:

```bash
python -m pip install -r requirements_modelado.txt
```

En Colab puedes usar `%pip install -r` seguido de la ruta a ese archivo. Después de instalar o cambiar versiones, reinicia la sesión.

En modo `revisar`, una diferencia de versiones genera una **advertencia** y queda registrada; por sí sola no detiene la revisión. Las dependencias necesarias deben estar disponibles y los controles de datos y resultados deben cumplirse. En modo `reentrenar`, una diferencia respecto de las versiones fijadas sí detiene el entrenamiento.

**Distinción entre entorno fijado y entorno registrado:** el archivo de dependencias exige NumPy 2.1.3, mientras que los manifiestos de calibración y pronósticos incluidos en `resultados/` registran NumPy 2.4.6. La evidencia original en niveles registra 2.1.3. Esta discrepancia se conserva y se declara; no se altera el registro de las ejecuciones ni se afirma que instalar el archivo de dependencias reproduzca todos sus valores exactamente. Revisar predicciones guardadas y repetir el entrenamiento son actividades distintas.

### Generar una nueva corrida de las extensiones

Para esa actividad, utiliza en **02 y 03** `MODO = 'reentrenar'` y una misma carpeta nueva, por ejemplo `CARPETA_RESULTADOS = 'resultados_nueva_corrida'`. En **04** conserva `MODO = 'revisar'`, pero indica esa misma carpeta nueva. Ejecuta 02 → 03 → 04 con las versiones exigidas y sin reutilizar una carpeta que ya contenga otra calibración.

Esta ruta recalibra y entrena **Δz y Ridge**, y calcula las referencias de la nueva corrida. Los modelos en **niveles** continúan reutilizando la evidencia v2 verificada: no es un entrenamiento integral de todas las estrategias desde cero. Los resultados incluidos se revisan con la ruta anterior; no hace falta reentrenar para leerlos.

## Qué se encontró en la corrida incluida

En las comparaciones siguientes, la cifra central es la **mediana de las diferencias pareadas de MASE por distrito** y los corchetes contienen el IC95 bootstrap distrital. Una diferencia negativa favorece al primer método de la comparación.

1. **Principal, E1:** con la configuración local compartida, Global en niveles tiene menor error que Local: **−0.081 [−0.099, −0.058] en Robo** y **−0.066 [−0.091, −0.046] en Hurto**. Global tiene menor MASE en 36 y 34 de los 43 distritos, respectivamente. Esto compara los mismos hiperparámetros por modelo; no iguala la capacidad total de 43 modelos Locales y un modelo Global.
2. **Contexto frente a referencias, E2:** Global en niveles con **H_L**, el brazo Global de E1, tiene diferencias frente a naïve-1 de **+0.024 [−0.008, +0.047]** y **+0.020 [−0.015, +0.050]**. Los intervalos incluyen cero; no muestran una ventaja clara ni demuestran equivalencia. Local tiene mayor error que naïve-1 según esta comparación. La sensibilidad Global con **H_G** debe leerse por separado: en Robo su diferencia frente a naïve-1 es **+0.031 [+0.011, +0.059]**.
3. **Sensibilidad y descripción pre/post, E3–E4:** la mediana Global − Local en niveles es negativa antes y después de noviembre de 2024: Robo **−0.114 → −0.054**; Hurto **−0.069 → −0.053**. El cambio se calcula primero en cada distrito. Su mediana es **+0.070 [−0.004, +0.122]** en Robo y **+0.019 [−0.031, +0.083]** en Hurto; no se interpreta como efecto causal del cambio de registro. En E3, Global sin Santa María del Mar se compara con Global con ese distrito, evaluando ambos sobre los mismos 42 distritos restantes. En Robo la diferencia es **+0.006 [+0.001, +0.012]**, mientras Wilcoxon–Holm da p ≈ 0.33: el intervalo de la mediana y la prueba de rangos responden preguntas diferentes.
4. **Exploratorio, E5–E7:** con Δz y **H_L5 compartida**, las diferencias Global − Local son **−0.005 [−0.019, +0.005]** en Robo y **−0.010 [−0.017, +0.001]** en Hurto. Global-Δz con H_L5 mejora a naïve-1 por márgenes pequeños: **−0.013 [−0.024, −0.003]**, mejor en 30 de 43 distritos de Robo; **−0.025 [−0.030, −0.018]**, mejor en 36 de 43 en Hurto. La configuración importa: con **H_G5 propia del Global, E5b**, Robo presenta mayor error que Local, **+0.045 [+0.034, +0.062]**; Hurto presenta **−0.011 [−0.023, −0.006]**. E5 no demuestra equivalencia ni, por comparación con E1, identifica un mecanismo.
5. **Post hoc, E8:** Global-Δz con H_L5 no muestra una ventaja clara frente a SES: **+0.003 [−0.006, +0.017]** en Robo y **+0.007 [−0.003, +0.011]** en Hurto. Las comparaciones de los modelos en niveles frente a SES tienen intervalos positivos. SES frente a naïve-1 presenta **−0.015 [−0.036, 0]** en Robo y **−0.026 [−0.047, −0.014]** en Hurto. Incluir cero no permite afirmar equivalencia.
6. **Post hoc descriptivo, E9–E10:** la ventaja Global − Local en niveles es mayor en 2023 (Robo **−0.131**; Hurto **−0.157**). En Robo se reduce cada año hasta **−0.014** en 2026; en Hurto los años siguientes presentan valores entre **−0.026 y −0.053**. **2026 solo incluye enero–julio.** Con Δz y H_L5, Global tiene menor error en 5 de los 15 distritos de menor volumen previo en Robo y en 6 de 15 en Hurto. Son descripciones de esta corrida, no explicaciones causales ni una ausencia de mejora en todos los distritos pequeños.

**Cronología de las decisiones:** la calibración selecciona hiperparámetros con los errores de 2022. Sin embargo, Δz y SES se incorporaron después de examinar la corrida principal en niveles; SES ya había sido examinado en un sondeo. Evaluarlos sobre el mismo periodo externo conserva su estatus exploratorio o post hoc. No deben presentarse como decisiones tomadas sin conocer resultados externos previos.

## Dónde está la evidencia

Las tablas y figuras del paper están en `resultados/04_resultados/paper/`. Los nombres siguientes corresponden a los archivos actuales; `*` representa el sufijo `robo` o `hurto`.

| Resultado que se quiere explicar | Tabla | Figura generada por Notebook 04 |
|---|---|---|
| E1 y E1b: Global − Local en niveles | `table2_global_vs_local.md` | `fig1_E1_*` |
| Comparación de métodos frente a naïve-1 | `table1_strategies.md`; tabla completa `tableA2_all_strategies.md` | `fig2_referencias_*` |
| Comparación frente a SES | Columnas A − SES de las tablas 1 y A2 | No hay figura específica; se interpreta la tabla. |
| E5 y E5b: Global − Local con Δz | `table2_global_vs_local.md` | `fig3_E5_distritos_*` muestra los 43 pares de **E5**; E5b se consulta en la tabla. |
| E3 y E4: sensibilidad y cambio pre/post | `tableA1_e4_e3.md` | `figA1_E4_*` describe los errores mensuales posteriores; no representa el contraste pre/post. |
| E10: evolución anual | `tableA3_annual.md` | `fig4_anual_*` |
| E5c: especificación Δz frente a niveles | `tableA4_dz_vs_levels.md` | No hay figura específica. |

El análisis de volumen E9 se consulta en `resultados/04_resultados/tablas/E9_volumen.csv`. No se le atribuye un gráfico que la notebook no produce.

Cada figura vigente tiene **PNG, SVG, CSV con sus datos y TXT con su descripción**. El texto se muestra al ejecutar la celda que la genera. Cada carpeta de figuras tiene un `indice_figuras.csv` con la notebook y la sección de origen. Hay **4 figuras de 02, 2 de 03 y 10 de 04**. Los PNG históricos `didactica_*.png`, fuera de esos índices, no se cargan ni se muestran en el flujo actual; no se usan como evidencia gráfica de esta entrega.

## Cómo interpretar y redactar las conclusiones

- Identifica siempre **target, formulación, configuración, periodo y referencia**. «Global tiene menor error» necesita indicar frente a qué estrategia y en qué comparación.
- Distingue la **mediana de diferencias por distrito** de la diferencia entre dos medianas de MASE. Las columnas de MASE y MAE de las tablas son resúmenes descriptivos; las columnas A − B son contrastes pareados.
- Un MASE menor que 1 no demuestra una mejora frente a naïve-1. La escala Q_d procede del naïve estacional de 2018–2022; la comparación con naïve-1 usa sus errores en los mismos meses evaluados.
- Los intervalos bootstrap, Wilcoxon y Holm no demuestran equivalencia. Holm ajusta los dos targets **dentro de cada contraste**, no el conjunto de todos los análisis.
- E1 es la comparación principal; E1b y E3 son sensibilidades; E4 es secundario y no causal; E5–E7 son exploratorios; E8–E10 son post hoc. E9–E10 tienen finalidad descriptiva.
- El cambio a Δz modifica la variable respuesta, las variables de entrada y la grilla de calibración. No permite atribuir una mejora únicamente a una de esas decisiones ni probar que XGBoost aporte no linealidad frente a Ridge.
- `ses_pendiente` es SES más media pendiente OLS en la escala logarítmica, una referencia ad hoc. No se denomina Theta estándar ni se afirma equivalencia con SES.
- El estudio evalúa denuncias registradas en los mismos distritos del entrenamiento. No sustenta afirmaciones sobre criminalidad real, riesgo individual, patrullaje ni generalización a distritos nuevos.
- El registro de decisiones es interno. No se presenta como prerregistro externo ni como prueba de una reproducción independiente del entrenamiento en otro equipo.

### Limitaciones que deben conservarse en el paper

**Decisiones exploratorias:** Δz y SES se examinaron después de observar la corrida principal. La selección de hiperparámetros con 2022 no elimina esa cronología. **Medición:** se estudian denuncias registradas, con un cambio de registro en noviembre de 2024 y una escala MASE fija que incluye 2020. **Inferencia:** el remuestreo distrital no resuelve los shocks comunes o la dependencia espacial; con 43 distritos los intervalos de medianas pueden ser discretos. **Alcance:** los datos corresponden a una ciudad, un snapshot y una evaluación temporal sobre distritos conocidos.

## Archivos y carpetas del paquete

| Ruta | Contenido y uso |
|---|---|
| `00. Datasets/` | Snapshot original utilizado por la auditoría. |
| `01. Auditoria SIDPOL/` | Panel, constantes MASE y metadatos de la auditoría cerrada. |
| `evidencia_registrada/` | Calibración y pronósticos originales en niveles; también conserva resúmenes recibidos de una corrida. Los resúmenes por sí solos no acreditan una reproducción independiente. |
| `resultados/02_calibracion/` | Configuración congelada, scores y predicciones internas; ejemplos y figuras de revisión en `figuras_notebook/`. |
| `resultados/03_pronosticos/` | Tabla de 44,290 pronósticos: 11 estrategias con 3,698 filas cada una y Global-42 con 3,612; ejemplos y figuras en `figuras_notebook/`. |
| `resultados/04_resultados/` | Tablas de análisis, tablas del paper, figuras e índice. |
| `sidpol.py` | Funciones de datos, modelos, comprobaciones, métricas y exportación de tablas. |
| `sidpol_figuras.py` | Guarda y muestra las figuras construidas en las celdas de las notebooks. |
| `GLOSARIO.md` | Definiciones y notación para interpretar las notebooks. |
| `requirements_modelado.txt` | Versiones exigidas por el modo de entrenamiento; véase la distinción de entornos anterior. |

Las rutas se muestran desde `SIDPOL_Notebooks` y, para las salidas incluidas, usan `CARPETA_RESULTADOS = 'resultados'`. Si creas otra corrida, sus salidas quedan bajo el nombre que hayas elegido. Este ZIP no incluye una carpeta `revision/`, scripts adicionales de verificación ni los documentos históricos de enmiendas; no se necesitan para ejecutar la ruta entregada.

## Alcance de esta actualización documental

Se actualizaron las explicaciones, fórmulas, referencias y notas de las tablas. Los datos, hiperparámetros, cálculos, pronósticos y figuras incluidos conservan sus valores. Notebook 01 permanece cerrada.

Los manifiestos conservan el entorno y la huella del código de sus ejecuciones originales. El manifiesto de 04 registra por separado esta revisión documental y las huellas actuales de las notas Markdown; no representa una nueva ejecución de modelos o del análisis. Los identificadores históricos, como `v0.8.5` en la configuración congelada y `v2` en la evidencia original, describen su procedencia, no la fecha de esta revisión.
