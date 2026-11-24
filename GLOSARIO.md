# Glosario SIDPOL

Estas definiciones corresponden al código y a los resultados incluidos. Robo y Hurto se estudian por separado. En las fórmulas, $d$ es el distrito, $t$ el mes y $y_{d,t}$ el número de denuncias registradas para un target fijo. La constante $Q_d$ también se llama `Q_j` en el código y en los archivos; hay una constante por combinación distrito–target.

| Término | Qué significa aquí |
|---|---|
| **Denuncias registradas** | Casos que figuran en el registro policial SIDPOL. No equivalen a toda la criminalidad ni a todos los hechos ocurridos. |
| **Target** | Variable que se pronostica: Robo o Hurto. Cada combinación de target y distrito forma una serie mensual. |
| **Distrito / UBIGEO** | Unidad geográfica del estudio: 43 distritos de Lima. UBIGEO es su código. |
| **Horizonte $h=1$** | Se pronostica el mes siguiente. Cada pronóstico utiliza el historial disponible hasta el mes anterior. |
| **Rolling-origin** | En la corrida que genera pronósticos, la ventana de entrenamiento crece un mes en cada paso y se vuelven a ajustar los modelos. Los controles temporales verifican el corte. En modo `revisar` se leen y comprueban pronósticos guardados. |
| **`cutoff` / `forecast_month` / `origin`** | `cutoff` es el último mes de entrenamiento y `forecast_month` el mes pronosticado. En los archivos de calibración, `origin` nombra el mes que se pronostica; su corte de entrenamiento es el mes anterior. |
| **Calibración (2022)** | Los hiperparámetros se seleccionan con los errores de enero–diciembre de 2022. La formulación Δz se incorporó tras revisar la corrida principal, por lo que sigue siendo exploratoria. |
| **Evaluación externa** | Evaluación temporal de enero de 2023 a julio de 2026: 43 meses sobre los mismos 43 distritos. No evalúa distritos nuevos. |
| **Local** | Un XGBoost por distrito: 43 modelos por target y por mes de entrenamiento. |
| **Global / pooling** | Un XGBoost por target y por mes de entrenamiento que utiliza conjuntamente las filas de los 43 distritos. |
| **Naïve-1** | Repite el último valor observado: $\widehat{y}_{d,t}=y_{d,t-1}$. Es una referencia de comparación. |
| **Naïve estacional** | Repite el valor de hace 12 meses: $\widehat{y}_{d,t}=y_{d,t-12}$. |
| **SES** | Suavizado exponencial simple sobre $z=\log(1+y)$. En cada mes, se elige $\alpha$ entre 20 valores de 0.05 a 1, minimizando el error cuadrático a un paso en la historia previa. Después se vuelve a la escala de denuncias. Con $\alpha=1$ reproduce naïve-1, salvo residuos numéricos. |
| **SES + ½ pendiente (`ses_pendiente`)** | Al nivel SES se suma la mitad de la pendiente OLS de la historia de $\log(1+y)$; después se vuelve a denuncias. Es una referencia ad hoc, no el método Theta estándar. |
| **Ridge** | Regresión lineal penalizada Global sobre Δz. Usa variables relativas y de calendario; no utiliza `lag_1` ni UBIGEO como entradas separadas. No es una comparación que cambie únicamente la no linealidad de XGBoost. |
| **$H_{\mathrm{L}}$ / $H_{\mathrm{G}}$ (`H_L` / `H_G`)** | Hiperparámetros elegidos con 2022 para Local y Global en niveles. E1 utiliza H_L en ambos: mismos hiperparámetros por modelo, sin igualar la capacidad total de 43 modelos Locales y uno Global. E1b utiliza H_G en Global y H_L en Local. |
| **$H_{\mathrm{L5}}$ / $H_{\mathrm{G5}}$ (`H_L5` / `H_G5`)** | Configuraciones elegidas con 2022 para Local y Global en Δz. E5 comparte H_L5; E5b usa H_G5 en Global. El sufijo 5 identifica la extensión, no un horizonte de cinco meses. |
| **$z=\log(1+y)$ (`log1p`)** | Transformación de la respuesta utilizada por los modelos en niveles. Reduce las diferencias de escala entre valores grandes y pequeños. |
| **$\Delta z$ (`dz`)** | Cambio respecto del mes anterior: $\Delta z_{d,t}=z_{d,t}-z_{d,t-1}$. Si el modelo predice un cambio cero, la retransformación devuelve el último valor observado, salvo precisión numérica. |
| **`lag_k`** | Valor de $z$ de hace $k$ meses, para el mismo target y distrito. |
| **Media móvil** | Promedio de los 3 o 6 valores anteriores de $z$. El código desplaza la serie con `shift(1)` para excluir el mes pronosticado. |
| **Variables relativas** | A los rezagos y medias móviles seleccionados se les resta `lag_1`. XGBoost-Δz también conserva `lag_1` y UBIGEO; Ridge utiliza un conjunto diferente. |
| **Retransformación (`expm1`)** | Vuelve de la escala logarítmica a denuncias con $\exp(z)-1$. En Δz se suma primero el cambio pronosticado al último $z$. Las salidas se recortan a cero como mínimo; no se aplica redondeo a entero ni corrección de sesgo (*smearing*). |
| **$Q_d$ (`Q_j`)** | Error absoluto medio del naïve estacional para cada serie distrito–target, usando 48 diferencias de 12 meses dentro de enero de 2018–diciembre de 2022. Se mantiene fijo en calibración y evaluación; no es una variable de entrenamiento. |
| **MASE del distrito** | Promedio del error absoluto dividido por $Q_d$ sobre los meses evaluados: 12 en calibración, 43 en la evaluación externa completa y la cantidad correspondiente en cada corte anual o pre/post. |
| **MASE menor que 1** | El error medio está por debajo de la escala histórica $Q_d$. No implica superar a naïve-1 ni al naïve estacional en el periodo externo: para ello se comparan errores sobre los mismos meses. |
| **MAE / RMSE** | MAE es la media del error absoluto; RMSE, la raíz de la media del error al cuadrado. Se calculan por distrito y periodo en la escala de denuncias. |
| **Score de calibración** | Mediana de los 43 MASE distritales de 2022 para una configuración. Primero se promedian los 12 errores de cada distrito; después se toma la mediana entre distritos. |
| **Diferencia pareada, A − B** | MASE de A menos MASE de B en cada distrito. Una diferencia negativa indica menor error de A. Se comparan los mismos distritos y meses. |
| **Mediana de diferencias** | Resumen central de las diferencias pareadas. Su signo indica la dirección del contraste y se interpreta junto con su intervalo y sus limitaciones. No equivale a restar dos medianas de MASE. |
| **IC95 bootstrap** | Intervalo aproximado de la mediana obtenido con 5,000 remuestreos de pares distritales y los percentiles 2.5 y 97.5. Trata los distritos como unidades de remuestreo independientes; los shocks comunes y la dependencia espacial limitan esa interpretación. Incluir cero no demuestra equivalencia. |
| **Empate numérico** | Se cuenta como cero una diferencia cuyo **valor absoluto** es menor que $10^{-12}$. Es una tolerancia de cálculo, no un margen de equivalencia práctica. Una diferencia de 0.003 no es un empate numérico; su interpretación requiere considerar magnitud e incertidumbre. |
| **Wilcoxon** | Prueba bilateral de rangos con signo sobre diferencias pareadas. Se aplica en E1, E1b, E3, E4 y los contrastes de E5. Los contrastes generales usan 43 distritos; E3 usa 42. No es la misma operación que el IC bootstrap de la mediana ni una prueba de equivalencia. |
| **Holm** | Ajuste de los valores p de Robo y Hurto dentro de cada contraste. No corrige conjuntamente todos los análisis exploratorios. |
| **Principal / sensibilidad / exploratorio / post hoc** | Principal: E1, con E2 como contexto frente a referencias. Sensibilidades: E1b y E3. Secundario no causal: E4. Exploratorios: E5–E7, incorporados tras revisar la corrida principal. Post hoc: E8–E10; E9–E10 tienen finalidad descriptiva. |
| **Registro interno** | Registro de las decisiones y de su cronología dentro del proyecto. No equivale a un prerregistro externo ni convierte una decisión exploratoria en confirmatoria. |
| **Santa María del Mar** | Distrito de volumen muy bajo. En Robo, un error de una denuncia representa aproximadamente 1.85 unidades de error escalado. E3 evalúa la sensibilidad a excluirlo del entrenamiento Global, manteniendo H_G y comparando ambos modelos sobre los mismos 42 distritos restantes. |
| **E4: cambio pre/post** | Diferencia entre el contraste Global − Local posterior y el anterior al cambio de registro, calculada primero en cada distrito. Después se resume con la mediana. No es un efecto causal. |
| **Banda Q1–Q3 / IQR** | Q1 y Q3 son los percentiles 25 y 75; el rango intercuartílico es $\mathrm{IQR}=Q_3-Q_1$. La banda de la curva E4 va de Q1 a Q3 entre distritos y no es un intervalo de confianza. |
| **Spearman $\rho$** | Correlación por rangos entre −1 y 1. E9 describe la asociación entre el volumen previo del distrito y la diferencia Global − Local. |
| **OLS** | Mínimos cuadrados ordinarios: ajusta la recta que minimiza la suma de errores al cuadrado. |
| ***Ad hoc*** | Construida para este estudio; no debe identificarse con otro método estándar sin verificar que sus fórmulas coinciden. |
| **Punto flotante** | Representación numérica de precisión finita; puede dejar residuos muy pequeños en operaciones matemáticamente equivalentes. |
| **Huella SHA-256** | Identificador calculado a partir de los bytes de un archivo. Se utiliza para detectar cambios y comprobar que se están leyendo los archivos previstos. |
| **Manifiesto (`manifest.json`)** | Registro del entorno y las huellas de una ejecución. Los manifiestos de figuras en modo `revisar` documentan una revisión; no sustituyen al registro del entrenamiento. |
| **Cadena 02 → 03 → 04** | La configuración y los scores se vinculan con los pronósticos, y estos con el análisis posterior mediante controles de huellas y de contenido. |
| **Corrida principal v2** | Ejecución de los modelos en niveles; sus predicciones se conservan en `evidencia_registrada/` y se reutilizan tras verificar sus huellas. |
| **Cambio de registro, noviembre de 2024** | Corte documentado en la auditoría. E4 compara enero de 2023–octubre de 2024 (22 meses) con noviembre de 2024–julio de 2026 (21 meses), sin atribuir causalmente las diferencias a ese cambio. |

Las fórmulas completas de transformación y score están en la Notebook 02; las de las referencias, en la 03; y las de Q_d, MASE, diferencias pareadas y E4, en la 04. Los nombres entre backticks son los identificadores utilizados por el código.
