import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np
import streamlit.components.v1 as components


def mostrar_animacion(ruta, obstaculos, ancho, alto, margen):
    """Representa el avance geométrico sobre una trayectoria."""
    ruta = np.asarray(ruta, dtype=float)

    # Elimina puntos consecutivos repetidos para interpolar sin divisiones por cero.
    conservar = np.concatenate(
        ([True], np.linalg.norm(np.diff(ruta, axis=0), axis=1) > 0.0)
    )
    puntos = ruta[conservar]

    longitudes = np.linalg.norm(np.diff(puntos, axis=0), axis=1)
    acumuladas = np.concatenate(([0.0], np.cumsum(longitudes)))
    longitud_total = float(acumuladas[-1])

    cantidad_cuadros = 80
    distancias = np.linspace(0.0, longitud_total, cantidad_cuadros)

    if longitud_total == 0.0:
        posiciones = np.repeat(puntos[:1], cantidad_cuadros, axis=0)
    else:
        posiciones = np.column_stack(
            (
                np.interp(distancias, acumuladas, puntos[:, 0]),
                np.interp(distancias, acumuladas, puntos[:, 1]),
            )
        )

    figura, eje = plt.subplots(figsize=(8, 5), dpi=90)
    figura.patch.set_facecolor("white")
    eje.set_facecolor("white")

    for obstaculo in obstaculos:
        centro = obstaculo["centro"]
        radio = obstaculo["radio"]

        eje.add_patch(
            Circle(
                centro,
                radio + margen,
                fill=False,
                edgecolor="#94A3B8",
                linestyle=":",
            )
        )

        eje.add_patch(
            Circle(
                centro,
                radio,
                facecolor="#E2E8F0",
                edgecolor="#64748B",
            )
        )

        eje.annotate(
            obstaculo["nombre"],
            centro,
            ha="center",
            va="center",
            fontsize=9,
        )

    eje.plot(
        puntos[:, 0],
        puntos[:, 1],
        "--",
        color="#94A3B8",
        linewidth=1.5,
        label="Trayectoria calculada",
    )

    eje.scatter(
        *puntos[0],
        color="#15803D",
        s=70,
        label="Inicio P",
        zorder=4,
    )

    eje.scatter(
        *puntos[-1],
        color="#7C3AED",
        marker="*",
        s=160,
        label="Destino Q",
        zorder=4,
    )

    recorrido, = eje.plot(
        [],
        [],
        color="#2563EB",
        linewidth=2.5,
        label="Recorrido completado",
    )

    dron, = eje.plot(
        [],
        [],
        marker="X",
        color="#EA580C",
        markersize=11,
        linestyle="None",
        label="Dron",
        zorder=6,
    )

    informacion = eje.text(
        0.02,
        0.98,
        "",
        transform=eje.transAxes,
        va="top",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.95,
        },
    )

    eje.set_xlim(0, ancho)
    eje.set_ylim(0, alto)
    eje.set_aspect("equal", adjustable="box")
    eje.set_xlabel("x (m)")
    eje.set_ylabel("y (m)")
    eje.set_title("Movimiento sobre la trayectoria calculada")
    eje.grid(True, alpha=0.2)

    eje.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    figura.subplots_adjust(bottom=0.25)

    def actualizar(indice):
        posicion = posiciones[indice]
        distancia = distancias[indice]

        # Conserva los vértices reales para dibujar fielmente los giros.
        vertices_recorridos = puntos[acumuladas <= distancia]
        trazo = np.vstack((vertices_recorridos, posicion))

        recorrido.set_data(trazo[:, 0], trazo[:, 1])
        dron.set_data([posicion[0]], [posicion[1]])

        porcentaje = 100.0
        if longitud_total > 0.0:
            porcentaje = 100.0 * distancia / longitud_total

        informacion.set_text(
            f"Posición: ({posicion[0]:.2f}, {posicion[1]:.2f}) m\n"
            f"Distancia recorrida: {distancia:.2f} m\n"
            f"Avance: {porcentaje:.1f}%"
        )

        return recorrido, dron, informacion

    animacion = FuncAnimation(
        figura,
        actualizar,
        frames=cantidad_cuadros,
        interval=100,
        blit=False,
        repeat=False,
    )

    try:
        contenido = animacion.to_jshtml(default_mode="once")
    finally:
        plt.close(figura)

    components.html(contenido, height=560, scrolling=True)