# Proyecto MT: umbral de Youden bayesiano

Proyecto de métodos numéricos de Alejandro Picón y Kevin Enrique Darghan. El programa calcula métricas de clasificación y estudia umbrales mediante bisección y Newton-Raphson.

## ¿Qué hace?

[`umbral_youden.py`](./umbral_youden.py) parte de una matriz de confusión fija (TP=9, FP=10, FN=18 y TN=175), calcula sensibilidad, especificidad, valores predictivos, Youden clásico y Youden bayesiano, y compara el umbral habitual `c = 0.5` con un umbral obtenido al resolver `g(c) = 0`. También explora las cinco preguntas de la propuesta, compara los dos métodos numéricos y crea tres gráficas.

**Alcance del resultado:** una matriz de confusión no contiene las puntuaciones individuales del clasificador, por lo que no permite reconstruir por sí sola el comportamiento real para todos los umbrales. El script genera puntuaciones sintéticas bajo un modelo binormal en la escala logit, calibrado para reproducir la matriz en `c = 0.5`. El parámetro `TAU = 1.6` es un supuesto, no se puede deducir de la matriz. Por eso, los umbrales y gráficas fuera del punto `0.5` son resultados del modelo supuesto, no estimaciones verificadas con observaciones individuales.

## Instalación y ejecución desde cero

Se requiere Python 3.11 o posterior; la ejecución de este proyecto se validó con Python 3.15.2.

### Linux o macOS

Desde la carpeta raíz del proyecto:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python umbral_youden.py
```

### Windows (PowerShell)

Desde la carpeta raíz del proyecto:

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\umbral_youden.py
```

Si PowerShell bloquea la activación del entorno, puede ejecutar los comandos de instalación y ejecución usando directamente `.\.venv\Scripts\python.exe`.

El programa imprime las métricas, iteraciones y respuestas en la terminal y crea estas imágenes dentro de `salidas/`:

- `convergencia.png`
- `matrices.png`
- `metricas_y_prevalencia.png`

Para repetir la ejecución, active el entorno virtual y vuelva a ejecutar el último comando. Para cambiar el escenario, edite las constantes documentadas al inicio de `umbral_youden.py`; `TAU`, en particular, es un supuesto que conviene analizar mediante sensibilidad.

## Respuestas de la propuesta y documentos de referencia

Las respuestas desarrolladas a las cinco preguntas están en [`RESPUESTAS_PROPUESTA.md`](./RESPUESTAS_PROPUESTA.md).

Los PDF originales se guardan en la carpeta local `privado/`, excluida por `.gitignore`; no se publican en este repositorio. La exclusión de Git no protege archivos que ya se hayan subido o enviado por otro medio. Para compartirlos entre equipos sin hacerlos públicos, use un repositorio privado o almacenamiento privado.
