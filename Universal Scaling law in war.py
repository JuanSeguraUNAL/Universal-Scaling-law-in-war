import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm


# Fraccionamiento
def fracc(n):
    DistF = n / np.sum(n)  # ----> Distribución de probabilidad para escoger el frente
    i = np.random.choice(np.arange(len(n)), p=DistF)  # Escoge un frente basado en la ponderación

    if n[i] > 1:  # ----> Solo fraccionar si el frente tiene al menos fuerza 2
        nuevos = np.ones(n[i], dtype=int)  # ----> Crea `n[i]` frentes de valor 1
        n = np.append(n, nuevos)  # ----> Añade los nuevos frentes de valor 1
        n = np.delete(n, i)  # ----> Elimina el frente original

    return n


# Combinación
def comn(n):
    DistF = n / np.sum(n)  # ----> Distribución de probabilidad para escoger los frentes
    i = np.random.choice(np.arange(len(n)), p=DistF)
    j = np.random.choice(np.arange(len(n)), p=DistF)

    # Asegurarse de que i y j sean distintos
    while j == i:
        j = np.random.choice(np.arange(len(n)), p=DistF)

    n[i] = n[i] + n[j]  # ----> Combina las fuerzas de los frentes
    n = np.delete(n, j)  # ----> Elimina uno de los frentes combinados

    return n


def simulacion(N, m, v):
    # N: Fuerza de ataque total
    # m: Número de iteraciones
    # v: Probabilidad de fragmentación

    ns = np.random.randint(1, N)  # ----> Número aleatorio entre 1 y N de unidades de ataque
    dist = np.random.multinomial(N, [1 / ns] * ns)  # ----> Distribución aleatoria inicial
    dist = dist[dist > 0]  # ----> Filtrar valores mayores que cero

    # Realizar las m iteraciones considerando probabilidad de fraccionamiento y combinación
    for i in tqdm(range(m), desc="Processing", unit="iteration"):
        if np.random.rand() > v and len(dist) > 1:
            dist = comn(dist)
        else:
            dist = fracc(dist)

    return dist


dist = simulacion(100000, 100000, 0.01)

np.savetxt('resultados.txt', dist)

# Imprimir la distribución final y la suma final
print("Distribución final:", dist)
print("Suma final:", np.sum(dist))