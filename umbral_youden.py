"""
Umbral de clasificación con Youden bayesiano: bisección y Newton-Raphson
=========================================================================
Parte 1: métricas de la matriz de confusión con c = 0,5 (modelo 1).
Parte 2: umbral c* por bisección y Newton sobre g(c) = 0, y modelo 2 con ese umbral.
Parte 3: matrices de confusión comparadas.
Parte 4: respuesta a las 5 preguntas del estudio.

SUPUESTO CLAVE (léelo): una matriz de confusión sola NO contiene los puntajes del
modelo. Para poder escribir g(c) se asume el modelo binormal de la presentación:
logit(p) ~ N(m0, tau^2) en sanos y N(m1, tau^2) en enfermos. m0 y m1 se calibran para
reproducir EXACTAMENTE Se y Sp de la matriz con c = 0,5; tau no se puede deducir de la
matriz y es un supuesto (ver TAU y el análisis de sensibilidad de la Parte 4).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import norm
from scipy.optimize import brentq
from rich.console import Console
from rich.table import Table

console = Console()

# ----------------------------------------------------------------------------
# CONFIGURACIÓN (cambia aquí)
# ----------------------------------------------------------------------------
# Matriz del artículo (Tabla 2). Filas = predicho, columnas = real.
#                 Real ausencia  Real presencia
# Pred. ausencia      TN=175         FN=18
# Pred. presencia     FP=10          TP=9
TN, FN, FP, TP = 175, 18, 10, 9

TAU = 1.6          # dispersión del logit(p) (supuesto, no sale de la matriz)
C0 = 0.5           # umbral habitual / punto de partida de Newton
A0, B0 = 0.01, 0.99  # intervalo inicial de la bisección
TOL = 1e-10
SE_MIN = 0.60      # P1: sensibilidad mínima aceptable
PPV_MIN = 0.60     # P4: VPP mínimo exigido
OUT_DIR = "salidas"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# PARTE 1: métricas desde la matriz
# ----------------------------------------------------------------------------
def metricas(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return dict(Se=se, Sp=sp, VPP=ppv, VPN=npv, CJ=se + sp - 1, BJ=ppv + npv - 1,
                P=(tp + fn) / n, Exactitud=(tp + tn) / n)


def nivel(v, tipo):
    if tipo == "prop":
        for lim, txt in [(0.5, "bajo"), (0.8, "moderado"), (0.9, "bueno")]:
            if v < lim:
                return txt
        return "alto"
    if tipo == "youden":
        return "débil" if v < 0.2 else "moderado" if v < 0.5 else "fuerte"
    if tipo == "prev":
        return "muy baja (desbalance fuerte)" if v < 0.2 else "baja" if v < 0.4 else "equilibrada"


EXPLICA = {
    "Se": ("Sensibilidad", "De los casos realmente positivos, cuántos detecta el modelo. "
           "Baja = deja pasar muchos enfermos.", "prop"),
    "Sp": ("Especificidad", "De los casos realmente negativos, cuántos descarta bien. "
           "Baja = muchas falsas alarmas.", "prop"),
    "VPP": ("Valor predictivo positivo", "Si el modelo dice 'positivo', qué tan probable es que "
            "de verdad lo sea. Depende de la prevalencia.", "prop"),
    "VPN": ("Valor predictivo negativo", "Si el modelo dice 'negativo', qué tan probable es que "
            "de verdad lo sea.", "prop"),
    "CJ": ("Youden clásico", "Se + Sp - 1. Qué tan bien separa las clases. 0 = azar, 1 = perfecto.",
           "youden"),
    "BJ": ("Youden bayesiano", "VPP + VPN - 1. Igual, pero mirando la confianza en cada predicción "
           "(incluye la prevalencia).", "youden"),
    "P": ("Prevalencia", "Proporción de casos realmente positivos en la muestra.", "prev"),
    "Exactitud": ("Exactitud (accuracy)", "Proporción total de aciertos. Engañosa si una clase es rara.",
                  "prop"),
}


def mostrar_metricas(m, titulo):
    t = Table(title=titulo, show_lines=True)
    t.add_column("Métrica", style="bold cyan")
    t.add_column("Valor", justify="right")
    t.add_column("Lectura", justify="center")
    t.add_column("¿Qué significa?", overflow="fold")
    for k, (nombre, expl, tipo) in EXPLICA.items():
        t.add_row(nombre, f"{m[k]:.3f}", nivel(m[k], tipo), expl)
    console.print(t)
    console.print("[dim]Rangos de 'Lectura' = referencia orientativa mía (<0,5 bajo; <0,8 moderado; "
                  "<0,9 bueno; ≥0,9 alto; Youden <0,2 débil, <0,5 moderado), no una norma.[/dim]\n")


# ----------------------------------------------------------------------------
# MODELO BINORMAL EN LA ESCALA logit(p)
# ----------------------------------------------------------------------------
def logit(c):
    return np.log(c / (1 - c))


def sigmoide(s):
    return 1 / (1 + np.exp(-s))


class ModeloBinormal:
    """Se(c), Sp(c), VPP(c), VPN(c), g(c) y g'(c) con m0, m1 calibrados a la matriz en c=0,5."""

    def __init__(self, se0, sp0, prev, tau):
        self.P, self.tau = prev, tau
        self.m1 = tau * norm.ppf(se0)     # Se(0,5) = Phi(m1/tau)
        self.m0 = -tau * norm.ppf(sp0)    # Sp(0,5) = Phi(-m0/tau)

    def se(self, c):
        return norm.sf((logit(c) - self.m1) / self.tau)

    def sp(self, c):
        return norm.cdf((logit(c) - self.m0) / self.tau)

    def ppv(self, c):
        a, b = self.P * self.se(c), (1 - self.P) * (1 - self.sp(c))
        return a / (a + b)

    def npv(self, c):
        a, b = (1 - self.P) * self.sp(c), self.P * (1 - self.se(c))
        return a / (a + b)

    def cj(self, c):
        return self.se(c) + self.sp(c) - 1

    def bj(self, c):
        return self.ppv(c) + self.npv(c) - 1

    def g(self, c):
        # g(c) = (1-P)[1-Phi((logit c - m0)/tau)] - P*Phi((logit c - m1)/tau)
        t = logit(c)
        return (1 - self.P) * norm.sf((t - self.m0) / self.tau) - self.P * norm.cdf((t - self.m1) / self.tau)

    def dg(self, c):
        # g'(c) = -[ (1-P) phi(z0) + P phi(z1) ] / ( tau * c * (1-c) )
        t = logit(c)
        z0, z1 = (t - self.m0) / self.tau, (t - self.m1) / self.tau
        return -((1 - self.P) * norm.pdf(z0) + self.P * norm.pdf(z1)) / (self.tau * c * (1 - c))


def reconstruir_casos(mod, tp, fp, fn, tn):
    """Puntajes p por caso, construidos por cuantiles para que la matriz en c=0,5 sea EXACTAMENTE
    la del artículo. Es una reconstrucción sintética: no son datos reales."""
    def q_tramo(n_abajo, n_arriba, prob_abajo):
        qa = (np.arange(1, n_abajo + 1) - 0.5) / max(n_abajo, 1) * prob_abajo
        qb = prob_abajo + (np.arange(1, n_arriba + 1) - 0.5) / max(n_arriba, 1) * (1 - prob_abajo)
        return np.concatenate([qa[:n_abajo], qb[:n_arriba]])

    s_neg = mod.m0 + mod.tau * norm.ppf(q_tramo(tn, fp, tn / (tn + fp)))
    s_pos = mod.m1 + mod.tau * norm.ppf(q_tramo(fn, tp, fn / (tp + fn)))
    p = sigmoide(np.concatenate([s_neg, s_pos]))
    y = np.concatenate([np.zeros(len(s_neg), int), np.ones(len(s_pos), int)])
    return p, y


def matriz(p, y, c):
    pred = (p >= c).astype(int)
    return (int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()),
            int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum()))


def texto_matriz(titulo, tp, fp, fn, tn):
    return [titulo,
            "                 Real +   Real -",
            f"  Predicho +   {tp:6d}   {fp:6d}",
            f"  Predicho -   {fn:6d}   {tn:6d}"]


def imprimir_matrices(*bloques):
    """Imprime varias matrices una al lado de la otra en un solo print."""
    filas = zip(*bloques)
    print("\n".join("      |      ".join(f"{l:<34}" for l in fila) for fila in filas))


# ----------------------------------------------------------------------------
# MÉTODOS NUMÉRICOS
# ----------------------------------------------------------------------------
def biseccion(f, a, b, tol=TOL, maxit=200):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) y f(b) tienen el mismo signo: no hay raíz garantizada en [a, b].")
    hist = []
    for _ in range(maxit):
        m = (a + b) / 2
        hist.append(m)
        fm = f(m)
        if fm == 0 or (b - a) / 2 < tol:
            break
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    return hist[-1], hist


def newton(f, c0, df=None, tol=TOL, maxit=50):
    """Newton-Raphson en c. Si df es None usa derivada numérica (diferencia central).
    Devuelve (raíz, historial, convergió)."""
    if df is None:
        df = lambda x: (f(x + 1e-7) - f(x - 1e-7)) / 2e-7
    c, hist = c0, [c0]
    for _ in range(maxit):
        d = df(c)
        if d == 0 or not np.isfinite(d):
            return c, hist, False
        c_new = c - f(c) / d
        if not (0 < c_new < 1):          # el paso se sale del dominio de c: Newton falla
            hist.append(c_new)
            return c_new, hist, False
        hist.append(c_new)
        if abs(c_new - c) < tol:
            return c_new, hist, True
        c = c_new
    return c, hist, False


# ============================================================================
# PARTE 1 — MODELO 1: umbral 0,5
# ============================================================================
console.rule("[bold]PARTE 1 · Modelo 1 (umbral c = 0,5)")
m1_ = metricas(TP, FP, FN, TN)
imprimir_matrices(texto_matriz("Matriz del artículo (c = 0,5)", TP, FP, FN, TN))
print()
mostrar_metricas(m1_, "Métricas del modelo 1 (c = 0,5)")

mod = ModeloBinormal(m1_["Se"], m1_["Sp"], m1_["P"], TAU)
p_casos, y_casos = reconstruir_casos(mod, TP, FP, FN, TN)
assert matriz(p_casos, y_casos, 0.5) == (TP, FP, FN, TN), "la reconstrucción no reproduce la matriz"

# ============================================================================
# PARTE 2 — MÉTODOS NUMÉRICOS
# ============================================================================
console.rule("[bold]PARTE 2 · Umbral c* con bisección y Newton-Raphson")
print(f"Supuestos: tau = {TAU}  ->  m0 = {mod.m0:.4f}, m1 = {mod.m1:.4f}")
print("g(c)  = (1-P)[1-Phi((logit c - m0)/tau)] - P*Phi((logit c - m1)/tau)")
print("g'(c) = -[(1-P)*phi((logit c - m0)/tau) + P*phi((logit c - m1)/tau)] / (tau*c*(1-c))")
print(f"        (siempre negativa: g es estrictamente decreciente -> raíz única)")
print(f"g(0.01) = {mod.g(A0):+.5f},  g(0.99) = {mod.g(B0):+.5f}  (cambio de signo: bisección válida)")

c_biseccion, hist_b = biseccion(mod.g, A0, B0)
c_newton, hist_n, ok_n = newton(mod.g, C0, mod.dg)
c_ref = brentq(mod.g, 1e-9, 1 - 1e-9, xtol=1e-15)  # referencia solo para medir errores

if not ok_n:
    console.print("[red]Newton NO convergió desde c0 = 0,5 con estos datos.[/red]")

t = Table(title="Primeras 10 iteraciones", show_lines=False)
for col in ["k", "Bisección c(k)", "Error", "Newton c(k)", "Error", "g'(c_k) Newton"]:
    t.add_column(col, justify="right")
for k in range(10):
    b = hist_b[k] if k < len(hist_b) else hist_b[-1]
    n_ = hist_n[k + 1] if k + 1 < len(hist_n) else hist_n[-1]
    d = mod.dg(hist_n[k]) if k < len(hist_n) else mod.dg(hist_n[-1])
    t.add_row(str(k + 1), f"{b:.8f}", f"{abs(b - c_ref):.2e}", f"{n_:.8f}",
              f"{abs(n_ - c_ref):.2e}", f"{d:.4f}")
console.print(t)
print(f"c_biseccion = {c_biseccion:.10f}  ({len(hist_b)} iteraciones, tol {TOL:g})")
print(f"c_newton    = {c_newton:.10f}  ({len(hist_n) - 1} iteraciones, tol {TOL:g})")
print(f"c* (referencia brentq) = {c_ref:.10f}")
print(f"g'(c_0 = {C0}) = {mod.dg(C0):.5f}   |   g'(c*) = {mod.dg(c_ref):.5f}\n")

# Gráfica de estabilización
fig1, ax = plt.subplots(1, 2, figsize=(12, 4.5))
nb = min(len(hist_b), 25)
ax[0].plot(range(1, nb + 1), hist_b[:nb], "o-", label="Bisección", color="tab:blue")
ax[0].plot(range(0, len(hist_n)), hist_n, "s-", label="Newton-Raphson", color="tab:red")
ax[0].axhline(c_ref, color="gray", ls="--", label=f"c* = {c_ref:.4f}")
ax[0].set(xlabel="Iteración k", ylabel="c(k)", title="Estabilización del umbral")
ax[0].legend()
ax[0].grid(alpha=.3)
err_b = np.maximum(np.abs(np.array(hist_b) - c_ref), 1e-16)
err_n = np.maximum(np.abs(np.array(hist_n) - c_ref), 1e-16)
ax[1].semilogy(range(1, len(err_b) + 1), err_b, "o-", label="Bisección", color="tab:blue")
ax[1].semilogy(range(0, len(err_n)), err_n, "s-", label="Newton-Raphson", color="tab:red")
ax[1].set(xlabel="Iteración k", ylabel="|c(k) - c*| (log)", title="Error por iteración")
ax[1].legend()
ax[1].grid(alpha=.3, which="both")
fig1.tight_layout()
fig1.savefig(f"{OUT_DIR}/convergencia.png", dpi=150)

# ============================================================================
# PARTE 3 — MODELO 2 con el umbral numérico y comparación
# ============================================================================
console.rule("[bold]PARTE 3 · Modelo 2 (umbral c*) y comparación")
res = {}
for nombre, c in [("c = 0,5", 0.5), ("c_biseccion", c_biseccion), ("c_newton", c_newton)]:
    res[nombre] = matriz(p_casos, y_casos, c)
iguales = res["c_biseccion"] == res["c_newton"]
print(f"¿Bisección y Newton producen la misma matriz? {'Sí' if iguales else 'NO'}\n")

imprimir_matrices(texto_matriz("Modelo 1 (c = 0,5)", *res["c = 0,5"]),
                  texto_matriz(f"Modelo 2 (c* = {c_biseccion:.4f})", *res["c_biseccion"]))
print()
m2_ = metricas(*res["c_biseccion"])
mostrar_metricas(m2_, f"Métricas del modelo 2 (c* = {c_biseccion:.4f})")

t = Table(title="Cómo trabajó el modelo: antes vs. después", show_lines=True)
for col in ["Métrica", "c = 0,5", "c*", "Cambio"]:
    t.add_column(col, justify="right" if col != "Métrica" else "left")
for k, (nombre, _, _) in EXPLICA.items():
    t.add_row(nombre, f"{m1_[k]:.3f}", f"{m2_[k]:.3f}", f"{m2_[k] - m1_[k]:+.3f}")
t.add_row("Errores (FP + FN)", str(res["c = 0,5"][1] + res["c = 0,5"][2]), str(res["c_biseccion"][1] + res["c_biseccion"][2]), "")
console.print(t)

fig2, axs = plt.subplots(1, 2, figsize=(10, 4.2))
colores_matriz = ListedColormap(["#d9f2df", "#f8d7da"])
for a_, (nombre, titulo) in zip(axs, [("c = 0,5", "Modelo 1 (c = 0,5)"), ("c_biseccion", f"Modelo 2 (c* = {c_biseccion:.3f})")]):
    tp, fp, fn, tn = res[nombre]
    mat = np.array([[tp, fp], [fn, tn]])
    aciertos = np.array([[0, 1], [1, 0]])
    a_.imshow(aciertos, cmap=colores_matriz, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a_.text(j, i, mat[i, j], ha="center", va="center", fontsize=16, color="black")
    a_.set_xticks([0, 1], ["Real +", "Real -"])
    a_.set_yticks([0, 1], ["Pred. +", "Pred. -"])
    a_.set_title(titulo)
fig2.legend(handles=[
    Patch(facecolor="#d9f2df", edgecolor="gray", label="Aciertos (VP / VN)"),
    Patch(facecolor="#f8d7da", edgecolor="gray", label="Errores (FP / FN)"),
], loc="lower center", ncol=2)
fig2.tight_layout(rect=(0, 0.1, 1, 1))
fig2.savefig(f"{OUT_DIR}/matrices.png", dpi=150)

# ============================================================================
# PARTE 4 — LAS 5 PREGUNTAS
# ============================================================================
console.rule("[bold]PARTE 4 · Preguntas del estudio")
grid = np.linspace(1e-4, 1 - 1e-4, 200001)

# --- P1 ---------------------------------------------------------------------
console.print("[bold]P1. Umbral que maximiza B_J con sensibilidad mínima, y diferencia con el Youden clásico[/bold]")
c_max = float(sigmoide(mod.m1 + mod.tau * norm.ppf(1 - SE_MIN)))   # Se(c) = SE_MIN
ok = grid <= c_max
c_bj = grid[ok][np.argmax(mod.bj(grid[ok]))]
c_cj = grid[np.argmax(mod.cj(grid))]
c_bj_libre = grid[np.argmax(mod.bj(grid))]
print(f"  B_J sin restricción: máximo en c = {c_bj_libre:.4f} con Se = {mod.se(c_bj_libre):.3f} "
      f"(degenerado: premia no detectar casi nada).")
t = Table(title=f"P1 · restricción Se ≥ {SE_MIN}")
for col in ["Criterio", "c", "Se", "Sp", "C_J", "B_J"]:
    t.add_column(col, justify="right" if col != "Criterio" else "left")
for nombre, c in [(f"max B_J con Se ≥ {SE_MIN}", c_bj), ("max Youden clásico C_J", c_cj),
                  ("c* (FP = FN)", c_ref), ("c = 0,5", 0.5)]:
    t.add_row(nombre, f"{c:.4f}", f"{mod.se(c):.3f}", f"{mod.sp(c):.3f}", f"{mod.cj(c):.3f}", f"{mod.bj(c):.3f}")
console.print(t)
print(f"  Diferencia de umbral (B_J restringido - Youden clásico) = {c_bj - c_cj:+.4f}\n")

# --- P2 ---------------------------------------------------------------------
console.print("[bold]P2. ¿Cómo cambia c* si cambia la prevalencia? (clasificador fijo)[/bold]")
prevs = [0.02, 0.05, 0.10, mod.P, 0.20, 0.30, 0.40, 0.50]
t = Table(title="P2 · c*(P) por bisección")
for col in ["P", "c*", "Se(c*)", "Sp(c*)", "iter."]:
    t.add_column(col, justify="right")
curva_P = []
for P_ in sorted(prevs):
    m_ = ModeloBinormal(m1_["Se"], m1_["Sp"], P_, TAU)
    c_, h_ = biseccion(m_.g, A0, B0)
    curva_P.append((P_, c_))
    t.add_row(f"{P_:.3f}" + (" (obs.)" if abs(P_ - mod.P) < 1e-12 else ""), f"{c_:.4f}",
              f"{m_.se(c_):.3f}", f"{m_.sp(c_):.3f}", str(len(h_)))
console.print(t)
print("  Nota: m0, m1 y tau se mantienen fijos; solo cambia P. Si reentrenas con otra\n"
      "  prevalencia, Se y Sp cambian y esta curva deja de valer.\n")

# --- P3 ---------------------------------------------------------------------
console.print("[bold]P3. ¿Para qué umbrales coinciden B_J y C_J?[/bold]")
dif = mod.bj(grid) - mod.cj(grid)
idx = np.where(np.sign(dif[:-1]) * np.sign(dif[1:]) < 0)[0]
t = Table(title="P3 · cruces B_J = C_J")
for col in ["c", "C_J = B_J", "FP − FN (% de n)", "TP − TN (% de n)", "Causa"]:
    t.add_column(col, justify="right" if col != "Causa" else "left")
for i in idx:
    c_ = brentq(lambda x: mod.bj(x) - mod.cj(x), grid[i], grid[i + 1], xtol=1e-14)
    P_, se_, sp_ = mod.P, mod.se(c_), mod.sp(c_)
    fpfn = (1 - P_) * (1 - sp_) - P_ * (1 - se_)
    tptn = P_ * se_ - (1 - P_) * sp_
    causa = "FP = FN" if abs(fpfn) < 1e-6 else "TP = TN" if abs(tptn) < 1e-6 else "otra"
    t.add_row(f"{c_:.4f}", f"{mod.cj(c_):.3f}", f"{100 * fpfn:+.4f}", f"{100 * tptn:+.4f}", causa)
console.print(t)
print("  B_J − C_J = (FN − FP)·[TP/((TP+FP)(TP+FN)) − TN/((TN+FN)(TN+FP))]: se anula con FP = FN\n"
      "  (el cruce que usa la presentación, g(c) = 0) o con TP = TN.\n")

# --- P4 ---------------------------------------------------------------------
console.print(f"[bold]P4. Umbral que garantiza VPP ≥ {PPV_MIN:.0%} y sensibilidad sacrificada[/bold]")
f_ppv = lambda c: mod.ppv(c) - PPV_MIN
c_hi = 1 - 1e-6
if f_ppv(1e-4) < 0 < f_ppv(c_hi):
    c_ppv, _ = biseccion(f_ppv, 1e-4, c_hi)
    t = Table(title=f"P4 · VPP ≥ {PPV_MIN:.0%}")
    for col in ["Umbral", "c", "Se", "Sp", "VPP"]:
        t.add_column(col, justify="right" if col != "Umbral" else "left")
    for nombre, c in [("c = 0,5", 0.5), ("c*", c_ref), (f"VPP = {PPV_MIN:.0%}", c_ppv)]:
        t.add_row(nombre, f"{c:.4f}", f"{mod.se(c):.3f}", f"{mod.sp(c):.3f}", f"{mod.ppv(c):.3f}")
    console.print(t)
    print(f"  Sensibilidad sacrificada: {mod.se(c_ref) - mod.se(c_ppv):.3f} respecto a c*  |  "
          f"{mod.se(0.5) - mod.se(c_ppv):+.3f} respecto a c = 0,5\n")
else:
    c_ppv = None
    print("  Ese VPP no es alcanzable con este clasificador y esta prevalencia.\n")

# --- P5 ---------------------------------------------------------------------
console.print("[bold]P5. ¿Qué método es más rápido y confiable?[/bold]")
res5 = {}   # (método, ecuación) -> [intentos, fallos, iteraciones]
sin_raiz = 0
for P_ in [0.02, 0.05, 0.10, 0.127, 0.20, 0.30, 0.50]:
    for tau_ in [0.8, 1.2, 1.6, 2.5, 4.0]:
        m_ = ModeloBinormal(m1_["Se"], m1_["Sp"], P_, tau_)
        for nombre_obj, f_, df_ in [("g(c)=0", m_.g, m_.dg),
                                    (f"VPP={PPV_MIN:.0%}", lambda c, m=m_: m.ppv(c) - PPV_MIN, None)]:
            try:
                _, h = biseccion(f_, A0, B0)
            except ValueError:
                sin_raiz += 1      # sin cambio de signo en [0,01; 0,99]: se omite el escenario completo
                continue
            r = res5.setdefault(("Bisección", nombre_obj), [0, 0, []])
            r[0] += 1
            r[2].append(len(h))
            for c0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
                _, h, conv = newton(f_, c0, df_)
                r = res5.setdefault(("Newton", nombre_obj), [0, 0, []])
                r[0] += 1
                r[1] += (not conv)
                if conv:
                    r[2].append(len(h) - 1)
t = Table(title="P5 · 7 prevalencias × 5 tau; Newton con 5 puntos de partida (0,1 a 0,9)")
for col in ["Método", "Ecuación", "Ejecuciones", "Fallos", "Iter. (media)", "Iter. (máx.)"]:
    t.add_column(col, justify="right" if col not in ("Método", "Ecuación") else "left")
for (met, ec), (n_, f_, its) in res5.items():
    t.add_row(met, ec, str(n_), str(f_), f"{np.mean(its):.1f}", str(max(its)))
console.print(t)
print(f"  Escenarios omitidos por no tener raíz en [{A0}; {B0}]: {sin_raiz}. "
      "Newton 'falla' si el paso sale de (0,1), no converge o la derivada se anula.")
print(f"  En el caso base: bisección {len(hist_b)} iteraciones vs. Newton {len(hist_n) - 1} (tol {TOL:g}).\n")

# --- Sensibilidad al supuesto de tau (la limitación más importante) ----------
console.print("[bold]Sensibilidad de c* al supuesto de tau (no identificable desde la matriz)[/bold]")
t = Table()
for col in ["tau", "c*", "Se(c*)", "Sp(c*)"]:
    t.add_column(col, justify="right")
for tau_ in [0.8, 1.2, 1.6, 2.5, 4.0]:
    m_ = ModeloBinormal(m1_["Se"], m1_["Sp"], m1_["P"], tau_)
    c_, _ = biseccion(m_.g, A0, B0)
    t.add_row(f"{tau_:.1f}", f"{c_:.4f}", f"{m_.se(c_):.3f}", f"{m_.sp(c_):.3f}")
console.print(t)

# Figura 3: curvas de las métricas y c*(P)
fig3, ax = plt.subplots(1, 2, figsize=(12, 4.5))
cc = np.linspace(0.005, 0.995, 800)
for nombre, f_ in [("Se", mod.se), ("Sp", mod.sp), ("VPP", mod.ppv), ("VPN", mod.npv), ("C_J", mod.cj), ("B_J", mod.bj)]:
    ax[0].plot(cc, f_(cc), label=nombre)
for c, lab, col in [(0.5, "0,5", "k"), (c_ref, "c*", "red"), (c_bj, "max B_J|Se", "green")]:
    ax[0].axvline(c, ls="--", color=col, alpha=.7, label=f"{lab} = {c:.3f}")
ax[0].set(xlabel="Umbral c", ylabel="Valor", title="Métricas vs. umbral (modelo binormal)")
ax[0].legend(fontsize=8, ncol=2)
ax[0].grid(alpha=.3)
ax[1].plot(*zip(*curva_P), "o-")
ax[1].axvline(mod.P, ls="--", color="gray", label=f"P observada = {mod.P:.3f}")
ax[1].set(xlabel="Prevalencia P", ylabel="c*", title="P2 · umbral c* según la prevalencia")
ax[1].legend()
ax[1].grid(alpha=.3)
fig3.tight_layout()
fig3.savefig(f"{OUT_DIR}/metricas_y_prevalencia.png", dpi=150)

print(f"\nFiguras guardadas en ./{OUT_DIR}/")
plt.show()
