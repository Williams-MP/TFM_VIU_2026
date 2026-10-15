# TFM_VIU_2026
Repositorio final del TFM

# Detección de anomalías y explicación mediante IA generativa de fallos en sistemas de streaming

Código del Trabajo Fin de Máster (Máster en Inteligencia Artificial, VIU, curso 2025-2026).

El trabajo combina dos detectores de anomalías basados en Transformer (TranAD y TranAD+) con un modelo de lenguaje de código abierto ejecutado en local (familia Qwen), sobre la telemetría real de un servicio IPTV. Se organiza en tres experimentos:

| Experimento | Pregunta | Notebooks |
|---|---|---|
| 1. Comparación de detectores | ¿Detecta TranAD+ alertas distintas de las de TranAD, más allá del azar del entrenamiento? | 02, 03, 04 |
| 2. Detección con el modelo de lenguaje | ¿Detecta mejor las anomalías el modelo cuando se le describe el comportamiento esperado del sistema? | 05, 06, 07 |
| 3. Explicación de las alertas | ¿Son las explicaciones del modelo fieles a los datos, útiles y legibles? | 08, 09 |

## Datos y confidencialidad

Los datos están sujetos a un acuerdo de confidencialidad y **no se incluyen en este repositorio**. Tampoco se incluyen los prompts completos, los nombres descriptivos de las variables ni las respuestas del modelo de lenguaje. Todo ello está en una carpeta privada de Google Drive (`privado/`), fuera del control de versiones.

En el código y en la memoria, las variables se identifican con nombres genéricos (S01ERR … S04L01, V01 y V02), y no se describe qué mide cada una.

Sin los datos originales se pueden ejecutar los notebooks 01 a 04 con datos sintéticos (ver más abajo). Las cifras no coincidirán con las de la memoria.

## Estructura del repositorio

```
notebooks/        Los 9 notebooks del trabajo, en orden de ejecución
src/              Módulos que usan los notebooks
  config.py         Rutas, nombres de las variables y dimensiones
  arquitectura_tranad.py, entrenamiento.py, checkpoints.py
                    Detectores: arquitectura, entrenamiento reproducible y checkpoints
  evaluacion.py     Puntuación de anomalía y umbral POT
  llm_lab/          Modelo de lenguaje: carga, prompts, ejecución, comprobaciones y estadísticas
databricks/       Documentación (sin código) de la preparación de los datos en Databricks
datos_sinteticos/ Generador de datos con el mismo formato que los reales
```

## Cómo ejecutarlo en Google Colab

1. **Datos.** Los notebooks buscan los datos en `MyDrive/TFM_VIU_2026/datos`. Si tienes acceso a la carpeta compartida, añade un acceso directo a ella en tu «Mi unidad» con el nombre `TFM_VIU_2026`. Para usar otra ruta, define la variable de entorno `TFM_RUTA_DATOS` antes de la primera celda.
2. **Abrir un notebook.** Desde Colab: *Archivo → Abrir cuaderno → GitHub*, y busca `Williams-MP/TFM_VIU_2026`. También puedes usar los enlaces de la tabla de abajo.
3. **Ejecutar.** *Entorno de ejecución → Ejecutar todas*. La primera celda monta Google Drive y clona el repositorio. Si el repositorio todavía es privado, te pedirá un token de GitHub con permiso de lectura. Si es público, deja el campo vacío.

Colab ya trae casi todas las librerías necesarias. La única que falta es `bitsandbytes`, que solo hace falta para ejecutar el modelo de lenguaje (notebooks 05 y 08). Esos notebooks la instalan solos y piden reiniciar la sesión una vez. Para ejecutar el código fuera de Colab, `requirements.txt` recoge las dependencias.

## Notebooks y tiempos de ejecución

Los notebooks se ejecutan en orden: cada uno lee lo que guardaron los anteriores. **Las ejecuciones largas solo son necesarias para repetir el trabajo desde cero.** Si los resultados ya están en la carpeta de datos, los notebooks los cargan y se saltan esa parte.

| Notebook | Qué hace | GPU | Tiempo con los resultados guardados | Tiempo desde cero |
|---|---|---|---|---|
| [01_preparacion_datos](https://colab.research.google.com/github/Williams-MP/TFM_VIU_2026/blob/main/notebooks/01_preparacion_datos.ipynb) | Separa los valores imposibles y escala las ventanas | No | Unos minutos | Unos minutos |
| [02_entrenamiento_detectores](https://colab.research.google.com/github/Williams-MP/TFM_VIU_2026/blob/main/notebooks/02_entrenamiento_detectores.ipynb) | Entrena TranAD y TranAD+ con 5 semillas cada uno | Sí (T4) | Unos minutos por modelo (solo recalcula los errores) | **≈ 4,5 h** (27 min por modelo). La sección 7, opcional, repite los 10 entrenamientos: otras 4,5 h |
| [03_analisis_detectores](https://colab.research.google.com/github/Williams-MP/TFM_VIU_2026/blob/main/notebooks/03_analisis_detectores.ipynb) | Índice de Jaccard, alertas de consenso y umbral POT | No | Minutos | Minutos |
| [04_experimento_1](https://colab.research.google.com/github/Williams-MP/TFM_VIU_2026/blob/main/notebooks/04_experimento_1.ipynb) | Figuras y tablas del Experimento 1 | No | Minutos | Minutos |
| [05_experimento_2](https://colab.research.google.com/github/Williams-MP/TFM_VIU_2026/blob/main/notebooks/05_experimento_2.ipynb) | Entradas y ejecuciones del modelo de lenguaje | Solo si `EJECUTAR = True` | Minutos | **≈ 25 min por modelo** sobre la prueba (160 consultas). Más con Qwen2.5-14B |
| [06_experimento_2_resultados](https://colab.research.google.com/github/Williams-MP/TFM_VIU_2026/blob/main/notebooks/06_experimento_2_resultados.ipynb) | Acierto, McNemar y AUROC | No | Minutos | Minutos |
| [07_experimento_2_estadisticas](https://colab.research.google.com/github/Williams-MP/TFM_VIU_2026/blob/main/notebooks/07_experimento_2_estadisticas.ipynb) | Aciertos por tipo de ventana de los tres modelos | No | Minutos | Minutos |
| [08_experimento_3](https://colab.research.google.com/github/Williams-MP/TFM_VIU_2026/blob/main/notebooks/08_experimento_3.ipynb) | Explicaciones de las alertas con Qwen3.5-9B | Solo en las secciones 4 y 5 | Minutos | **≈ 2 h** en una L4 (unos 100 s por ventana, 69 ventanas) |
| [09_experimento_3_resultados](https://colab.research.google.com/github/Williams-MP/TFM_VIU_2026/blob/main/notebooks/09_experimento_3_resultados.ipynb) | Comprobaciones automáticas, puntuaciones y H₃ | No | Minutos | Minutos |

Los notebooks 05 a 09 necesitan la carpeta `privado/`.

Sobre las ejecuciones largas:

- **02:** antes de ejecutarlo hay que reiniciar el entorno de Colab, para que la configuración determinista de la GPU tenga efecto. Si Colab se corta, al volver a ejecutarlo continúa desde el último checkpoint. La sección 7 no tiene interruptor: con *Ejecutar todas* también se ejecuta, y solo se salta si `detectores_reproducibilidad/` ya está completa. Para revisar los resultados no hace falta ejecutar este notebook. El 03 trabaja con los errores ya guardados.
- **05 y 08:** por defecto no ejecutan el modelo. Usan las ejecuciones de la memoria, ya guardadas. Para repetirlas hay que activarlo de forma explícita en la celda de parámetros. Una L4 es más rápida que una T4, pero las dos sirven.

## Estructura de la carpeta de datos

```
datos/
  train/ val/ test/      Ventanas exportadas desde Databricks (Parquet)
  preparados/            Ventanas escaladas y escaladores (notebook 01)
  resultados/
    detectores/          Checkpoints y errores de los 10 modelos (02)
    detectores_reproducibilidad/   Repetición de los entrenamientos (02, sección 7)
    analisis_detectores/ Alertas de consenso, POT y Jaccard (03)
    experimento_1/       Figuras y tablas (04)
    experimento_2/       Cifras, sin textos del modelo (06, 07)
    experimento_3/       Cifras, sin textos del modelo (09)
  privado/               Confidencial: no se publica
    nombres_senales.json Nombres descriptivos que recibe el modelo de lenguaje
    prompts/             Textos de los prompts
    experimento_2/       Entradas, referencia del autor y respuestas del modelo
    experimento_3/       Respuestas del modelo, informes y puntuaciones
```

**Formato de entrada:** un fichero `ventanas.parquet` en cada una de `train/`, `val/` y `test/`. Hay una fila por ventana, con las columnas `device_pseudo_id`, `session_id`, `window_id`, `secuencia` (12 reportes × 13 variables), `inicio_ventana` y `fin_ventana`. Los identificadores de los dispositivos están seudonimizados.

## Datos sintéticos

`datos_sinteticos/generar_datos_sinteticos.py` genera ventanas aleatorias con el mismo formato que las reales. Así se puede comprobar que los notebooks 01 a 04 funcionan de principio a fin. Por ejemplo, en Colab: genera los datos con `--salida /content/datos_sinteticos` y define `TFM_RUTA_DATOS=/content/datos_sinteticos` antes de la primera celda.

## Reproducibilidad

- **Detectores:** las semillas (0, 7, 42, 123 y 777) controlan los pesos iniciales, el orden de las ventanas y el *dropout*, y PyTorch se configura en modo determinista. Dos entrenamientos con la misma semilla dan los mismos pesos (apartado 5.1.1 de la memoria).
- **Modelo de lenguaje:** se decodifica sin muestreo, así que la misma entrada da la misma salida. Cada ejecución guarda la huella SHA-256 del prompt, y los notebooks comprueban que coincide con la de la memoria antes de usar una ejecución.
- **Cifras de la memoria:** los notebooks 03, 04 y 09 comparan cada cifra calculada con la de la memoria y la marcan como `coincide` o `distinto`.

Entorno: Google Colab, Python 3.12 y PyTorch 2.9. GPU NVIDIA Tesla T4 para los detectores y el Experimento 2, y NVIDIA L4 para el Experimento 3.

## Autor

Williams M.
