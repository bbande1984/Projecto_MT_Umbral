# Respuestas a las preguntas de la propuesta

## Alcance y datos usados

Estas respuestas corresponden a la ejecución de `umbral_youden.py` con la matriz `TP=9`, `FP=10`, `FN=18`, `TN=175`, prevalencia observada `P=27/212 ≈ 0.1274` y `TAU=1.6`. La matriz permite calcular directamente las métricas en `c=0.5`, pero no contiene las puntuaciones de cada observación. Para estudiar otros umbrales, el programa construye puntuaciones sintéticas bajo un modelo binormal en la escala logit. Por ello, los resultados para `c ≠ 0.5` dependen del supuesto `TAU` y no son una evaluación empírica de las puntuaciones originales.

El programa usa `c*` para la raíz de `g(c)=0`, que iguala las proporciones esperadas de falsos positivos y falsos negativos. Ese umbral no es el máximo de Youden bayesiano de la pregunta 1; son criterios distintos.

El documento de análisis PDF guardado en la carpeta privada debe cotejarse antes de reutilizar sus cifras: si emplea otro objetivo de optimización, sus umbrales no son directamente comparables con los de esta ejecución.

## 1. Máximo de Youden bayesiano con sensibilidad mínima

Con el requisito `Se ≥ 0.60`, el máximo de Youden bayesiano se obtiene aproximadamente en:

| Criterio | Umbral `c` | Sensibilidad | Especificidad | `C_J` | `B_J` |
|---|---:|---:|---:|---:|---:|
| Máximo `B_J`, sujeto a `Se ≥ 0.60` | 0.2508 | 0.600 | 0.822 | 0.422 | 0.263 |
| Máximo Youden clásico `C_J` | 0.1638 | 0.722 | 0.722 | 0.443 | 0.221 |

El umbral restringido de `B_J` es aproximadamente `0.0869` mayor que el que maximiza `C_J`. Sin imponer un mínimo de sensibilidad, el máximo de `B_J` puede favorecer un umbral extremo que detecta muy pocos positivos; por eso se reporta la restricción.

## 2. Cambio del umbral con la prevalencia

Manteniendo fijos los parámetros del clasificador (`m0`, `m1` y `TAU`) y cambiando solo la prevalencia, la raíz de `g(c)=0` varía así:

| Prevalencia `P` | Umbral `c*` |
|---:|---:|
| 0.020 | 0.6953 |
| 0.050 | 0.5660 |
| 0.100 | 0.4509 |
| 0.127 (observada) | 0.4085 |
| 0.200 | 0.3281 |
| 0.300 | 0.2555 |
| 0.400 | 0.2040 |
| 0.500 | 0.1638 |

En este escenario, al aumentar la prevalencia, el umbral que iguala las proporciones esperadas de falsos positivos y falsos negativos disminuye. Es una comparación teórica con clasificador fijo; si el clasificador se reentrena o recalibra, la curva puede cambiar.

## 3. Umbrales donde coinciden `B_J` y `C_J`

En el rango explorado se encuentran dos cruces:

- `c ≈ 0.0138`: la igualdad se produce cuando las proporciones esperadas de verdaderos positivos y verdaderos negativos coinciden.
- `c ≈ 0.4085`: coincide con la raíz de `g(c)=0`, donde las proporciones esperadas de falsos positivos y falsos negativos coinciden.

El cruce de `0.4085` es el que el código denomina `c*`; no debe confundirse con el umbral de la pregunta 1.

## 4. Umbral para garantizar `VPP ≥ 0.60`

El umbral modelado para alcanzar `VPP=0.60` es aproximadamente `c=0.6987`. En ese punto, la sensibilidad es `0.169`. Frente a la sensibilidad `0.421` en `c*=0.4085`, se sacrifican aproximadamente `0.252` puntos de sensibilidad; frente a `c=0.5` (`Se=0.333`), la reducción es de `0.164`.

La garantía es condicional al modelo y a la prevalencia introducidos. No garantiza ese VPP en otra población sin validar o actualizar el modelo.

## 5. Método más rápido y confiable

El programa compara bisección y Newton-Raphson en varios valores de prevalencia, `TAU` y puntos iniciales. En el barrido ejecutado:

| Ecuación | Método | Ejecuciones | Fallos | Iteraciones medias | Máximo |
|---|---|---:|---:|---:|---:|
| `g(c)=0` | Bisección | 35 | 0 | 34.0 | 34 |
| `g(c)=0` | Newton-Raphson | 175 | 45 | 5.9 | 10 |
| `VPP=0.60` | Bisección | 30 | 0 | 34.0 | 34 |
| `VPP=0.60` | Newton-Raphson | 150 | 51 | 5.1 | 8 |

Bisección converge cuando el intervalo contiene un cambio de signo y mantiene la raíz acotada; suele necesitar más iteraciones. Newton-Raphson puede llegar en menos iteraciones cuando el punto inicial es adecuado, pero puede salir del dominio `(0,1)` o no converger. Por tanto, no hay un ganador universal: para robustez se prefiere bisección con un intervalo válido; para rapidez puede convenir Newton cuando se verifica su convergencia. En el caso base, bisección requiere 34 iteraciones y Newton-Raphson 5.

Las medias de iteraciones de Newton-Raphson se calculan solo sobre las ejecuciones que convergieron; los fallos se contabilizan por separado.

## Limitaciones que deben acompañar la interpretación

- La matriz usada aparece en el documento de referencia como una clasificación basada en índices multiespectrales; la propuesta, en cambio, plantea un ejemplo de resistencia a la insulina en mujeres. La procedencia y pertinencia de la matriz deben confirmarse antes de atribuir estos resultados a ese estudio.
- `TAU=1.6` no se estima a partir de la matriz. El propio programa incluye una sensibilidad de `c*` a ese supuesto.
- La matriz sintética generada reproduce los conteos de `c=0.5`, pero no recupera los puntajes ni las predicciones individuales originales.
- La ecuación y el criterio de optimización de cada umbral deben nombrarse explícitamente: maximizar `B_J` con una restricción de sensibilidad, igualar falsos positivos y falsos negativos, y alcanzar un VPP objetivo son problemas diferentes.
