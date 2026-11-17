import heapq
import math

import numpy as np


def segmento_libre(inicio, destino, obstaculos, margen):
    """Comprueba el segmento completo frente a todos los discos."""
    vector = destino - inicio
    longitud_cuadrada = float(np.dot(vector, vector))

    for obstaculo in obstaculos:
        centro = obstaculo["centro"]

        if longitud_cuadrada == 0.0:
            punto_cercano = inicio
        else:
            parametro = np.dot(centro - inicio, vector) / longitud_cuadrada
            parametro = float(np.clip(parametro, 0.0, 1.0))
            punto_cercano = inicio + parametro * vector

        distancia = float(np.linalg.norm(centro - punto_cercano))
        radio_protegido = obstaculo["radio"] + margen

        if distancia <= radio_protegido + 1e-8:
            return False

    return True


def calcular_ruta(inicio, destino, obstaculos, ancho, alto, margen):
    """Busca la ruta más corta en un grafo de puntos candidatos."""
    if segmento_libre(inicio, destino, obstaculos, margen):
        return np.array([inicio, destino])

    puntos = [inicio.copy(), destino.copy()]
    cantidad_por_obstaculo = 32

    for obstaculo in obstaculos:
        # Este radio permite que los segmentos entre puntos vecinos
        # rodeen el disco ampliado por el margen de separación.
        radio_candidatos = (
            obstaculo["radio"] + margen + 1e-6
        ) / math.cos(math.pi / cantidad_por_obstaculo)

        for angulo in np.linspace(
            0.0,
            2.0 * math.pi,
            cantidad_por_obstaculo,
            endpoint=False,
        ):
            punto = obstaculo["centro"] + radio_candidatos * np.array(
                [math.cos(angulo), math.sin(angulo)]
            )

            dentro_entorno = (
                0.0 <= punto[0] <= ancho
                and 0.0 <= punto[1] <= alto
            )

            if dentro_entorno and segmento_libre(
                punto, punto, obstaculos, margen
            ):
                puntos.append(punto)

    puntos = np.array(puntos)
    conexiones = [[] for _ in puntos]

    for origen in range(len(puntos)):
        for destino_indice in range(origen + 1, len(puntos)):
            if segmento_libre(
                puntos[origen],
                puntos[destino_indice],
                obstaculos,
                margen,
            ):
                distancia = float(
                    np.linalg.norm(puntos[destino_indice] - puntos[origen])
                )
                conexiones[origen].append((destino_indice, distancia))
                conexiones[destino_indice].append((origen, distancia))

    # Dijkstra busca el camino de menor longitud en el grafo.
    distancias = [math.inf] * len(puntos)
    anteriores = [None] * len(puntos)
    distancias[0] = 0.0
    pendientes = [(0.0, 0)]

    while pendientes:
        distancia_actual, actual = heapq.heappop(pendientes)

        if distancia_actual > distancias[actual]:
            continue

        if actual == 1:
            indices = [1]

            while indices[-1] != 0:
                indices.append(anteriores[indices[-1]])

            indices.reverse()
            return puntos[indices]

        for vecino, longitud in conexiones[actual]:
            nueva_distancia = distancia_actual + longitud

            if nueva_distancia < distancias[vecino]:
                distancias[vecino] = nueva_distancia
                anteriores[vecino] = actual
                heapq.heappush(pendientes, (nueva_distancia, vecino))

    return None
