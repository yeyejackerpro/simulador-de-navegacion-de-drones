import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import streamlit as st
from rutas import calcular_ruta, segmento_libre
from animacion import mostrar_animacion

st.set_page_config(
    page_title="Simulador de navegación de drones",
    page_icon="🛩️",
    layout="wide",
)


def analizar_segmento(inicio, destino, centro):
    """Calcula el punto del segmento más cercano al centro."""
    vector = destino - inicio
    longitud_cuadrada = float(np.dot(vector, vector))

    if longitud_cuadrada == 0.0:
        return None, inicio.copy(), float(np.linalg.norm(centro - inicio))

    parametro = float(np.dot(centro - inicio, vector) / longitud_cuadrada)
    parametro_segmento = float(np.clip(parametro, 0.0, 1.0))
    punto_cercano = inicio + parametro_segmento * vector
    distancia = float(np.linalg.norm(centro - punto_cercano))

    return parametro, punto_cercano, distancia


st.title("Simulador de navegación de drones")
st.write(
    "Planificación de trayectorias en un entorno bidimensional "
    "mediante geometría vectorial."
)
st.info(
    "Todas las longitudes se expresan en metros (m). "
    "El dron se representa mediante un punto."
)

configuracion, simulacion, calculos = st.tabs(
    ["Configuración del escenario", "Simulación", "Cálculos geométricos"]
)

with configuracion:
    st.subheader("Dimensiones del entorno")
    izquierda, derecha = st.columns(2)

    with izquierda:
        ancho = st.number_input(
            "Ancho del entorno (m)",
            min_value=1.0,
            value=10.0,
            step=1.0,
            key="ancho",
        )

    with derecha:
        alto = st.number_input(
            "Alto del entorno (m)",
            min_value=1.0,
            value=8.0,
            step=1.0,
            key="alto",
        )

    st.caption("El entorno comienza en el origen (0, 0).")

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader("Posición inicial P")
        inicio_x = st.number_input(
            "Coordenada x de P (m)",
            min_value=0.0,
            value=2.0,
            step=0.1,
            key="inicio_x",
        )
        inicio_y = st.number_input(
            "Coordenada y de P (m)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key="inicio_y",
        )

    with derecha:
        st.subheader("Destino Q")
        destino_x = st.number_input(
            "Coordenada x de Q (m)",
            min_value=0.0,
            value=6.0,
            step=0.1,
            key="destino_x",
        )
        destino_y = st.number_input(
            "Coordenada y de Q (m)",
            min_value=0.0,
            value=4.0,
            step=0.1,
            key="destino_y",
        )

    inicio = np.array([inicio_x, inicio_y], dtype=float)
    destino = np.array([destino_x, destino_y], dtype=float)

    st.subheader("Separación respecto a los obstáculos")

    margen = st.number_input(
        "Margen mínimo de separación (m)",
        min_value=0.01,
        value=0.10,
        step=0.05,
        key="margen",
    )

    st.caption(
        "El planificador amplía cada radio con este margen. "
        "Todos los segmentos del recorrido deben quedar fuera "
        "de los discos ampliados."
    ) 
    st.subheader("Obstáculos circulares")
    st.write(
        "Edita las celdas para cambiar el centro y el radio. "
        "Puedes agregar filas al final de la tabla y seleccionar "
        "filas para eliminarlas."
    )

    tabla_inicial = pd.DataFrame(
        {
            "Centro x (m)": [3.0],
            "Centro y (m)": [3.0],
            "Radio (m)": [1.0],
        }
    )

    tabla = st.data_editor(
        tabla_inicial,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Centro x (m)": st.column_config.NumberColumn(
                min_value=0.0,
                step=0.1,
                required=True,
            ),
            "Centro y (m)": st.column_config.NumberColumn(
                min_value=0.0,
                step=0.1,
                required=True,
            ),
            "Radio (m)": st.column_config.NumberColumn(
                min_value=0.001,
                step=0.1,
                required=True,
            ),
        },
        key="tabla_obstaculos",
    )

    errores = []

    if not np.all(np.isfinite([ancho, alto, *inicio, *destino])):
        errores.append("Las dimensiones y coordenadas deben ser finitas.")

    if inicio_x > ancho or destino_x > ancho:
        errores.append("Las coordenadas x de P y Q deben estar dentro del entorno.")

    if inicio_y > alto or destino_y > alto:
        errores.append("Las coordenadas y de P y Q deben estar dentro del entorno.")

    obstaculos = []

    for numero, fila in enumerate(tabla.itertuples(index=False, name=None), 1):
        valores = pd.to_numeric(pd.Series(fila), errors="coerce").to_numpy(
            dtype=float
        )

        if not np.all(np.isfinite(valores)):
            errores.append(
                f"Completa el centro y el radio del obstáculo O{numero}."
            )
            continue

        centro_x, centro_y, radio = valores

        if radio <= 0.0:
            errores.append(f"El radio de O{numero} debe ser positivo.")
            continue

        if (
            centro_x - radio < 0.0
            or centro_x + radio > ancho
            or centro_y - radio < 0.0
            or centro_y + radio > alto
        ):
            errores.append(
                f"El disco completo de O{numero} debe quedar dentro del entorno."
            )
            continue

        obstaculos.append(
            {
                "nombre": f"O{numero}",
                "centro": np.array([centro_x, centro_y], dtype=float),
                "radio": float(radio),
            }
        )

    if errores:
        for error in errores:
            st.error(error)
    else:
        st.success("Los datos del escenario son válidos.")


# Tolerancia para clasificar contactos numéricos con el borde.
tolerancia = 1e-8
resultados = []
extremos_ocupados = []

if not errores:
    for obstaculo in obstaculos:
        centro = obstaculo["centro"]
        radio = obstaculo["radio"]

        parametro, punto_cercano, distancia = analizar_segmento(
            inicio, destino, centro
        )

        diferencia = distancia - radio

        if diferencia > tolerancia:
            estado = "Libre"
        elif diferencia < -tolerancia:
            estado = "Intersección interior"
        else:
            estado = "Contacto con el borde"

        resultados.append(
            {
                "Obstáculo": obstaculo["nombre"],
                "Radio (m)": radio,
                "t₀": parametro,
                "Kx (m)": float(punto_cercano[0]),
                "Ky (m)": float(punto_cercano[1]),
                "Distancia mínima (m)": distancia,
                "Distancia menos radio (m)": diferencia,
                "Estado": estado,
            }
        )

        if np.linalg.norm(inicio - centro) <= radio + tolerancia:
            extremos_ocupados.append(
                f"P está dentro o en el borde de {obstaculo['nombre']}."
            )

        if np.linalg.norm(destino - centro) <= radio + tolerancia:
            extremos_ocupados.append(
                f"Q está dentro o en el borde de {obstaculo['nombre']}."
            )

    directo_libre = all(
        resultado["Estado"] == "Libre" for resultado in resultados
    )

with simulacion:
    st.subheader("Escenario y conexión directa")

    if errores:
        st.warning("Corrige los datos en Configuración del escenario.")
    else:
        figura, eje = plt.subplots(figsize=(9, 6))
        figura.patch.set_facecolor("white")
        eje.set_facecolor("white")

        color_segmento = "#2563EB" if directo_libre else "#DC2626"

        eje.plot(
            [inicio_x, destino_x],
            [inicio_y, destino_y],
            "--",
            color=color_segmento,
            linewidth=2,
            label="Segmento directo P–Q",
        )

        for obstaculo, resultado in zip(obstaculos, resultados):
            centro = obstaculo["centro"]
            bloquea = resultado["Estado"] != "Libre"

            disco = Circle(
                centro,
                obstaculo["radio"],
                facecolor="#FEE2E2" if bloquea else "#E2E8F0",
                edgecolor="#B91C1C" if bloquea else "#64748B",
                linewidth=1.5,
                alpha=0.7,
            )
            eje.add_patch(disco)
            eje.plot(*centro, ".", color="#334155")
            eje.annotate(
                obstaculo["nombre"],
                centro,
                xytext=(5, 6),
                textcoords="offset points",
            )

        eje.scatter(
            *inicio,
            color="#15803D",
            s=100,
            label="Inicio P",
            zorder=4,
        )
        eje.scatter(
            *destino,
            color="#7C3AED",
            marker="*",
            s=200,
            label="Destino Q",
            zorder=5,
        )

        eje.set_xlim(0, ancho)
        eje.set_ylim(0, alto)
        eje.set_aspect("equal", adjustable="box")
        eje.set_xlabel("x (m)")
        eje.set_ylabel("y (m)")
        eje.set_title("Comprobación geométrica del segmento directo")
        eje.grid(True, alpha=0.25)
        eje.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=False,
        )

        st.pyplot(figura, use_container_width=True)
        plt.close(figura)

        for mensaje in extremos_ocupados:
            st.error(mensaje)

        if extremos_ocupados:
            st.warning(
                "Debes ubicar P y Q fuera de todos los obstáculos "
                "para planificar una trayectoria."
            )
        elif np.array_equal(inicio, destino):
            st.info("P y Q coinciden. El dron ya está en el destino.")
        elif directo_libre:
            st.success("El segmento directo está libre de obstáculos.")
        else:
            st.warning(
                "El segmento directo está bloqueado. "
                "Será necesario calcular un recorrido alternativo."
            )

with calculos:
    st.subheader("Desplazamiento y distancia directa")

    if errores:
        st.warning("Corrige los datos en Configuración del escenario.")
    else:
        desplazamiento = destino - inicio
        distancia_directa = float(np.linalg.norm(desplazamiento))

        st.latex(r"\overrightarrow{PQ}=(x_Q-x_P,\;y_Q-y_P)")
        st.write(
            f"Vector de desplazamiento: "
            f"({desplazamiento[0]:.3f}, {desplazamiento[1]:.3f}) m"
        )

        st.latex(r"d(P,Q)=\sqrt{(x_Q-x_P)^2+(y_Q-y_P)^2}")
        st.metric("Distancia directa", f"{distancia_directa:.3f} m")

        if distancia_directa > 0.0:
            direccion = desplazamiento / distancia_directa
            st.write(
                f"Vector unitario: "
                f"({direccion[0]:.4f}, {direccion[1]:.4f})"
            )
            st.caption("El vector unitario es adimensional.")
        else:
            st.info("El desplazamiento nulo no puede normalizarse.")

        st.subheader("Distancia mínima a los obstáculos")

        st.latex(
            r"t_0=\frac{(C-P)\cdot(Q-P)}{\|Q-P\|^2}"
        )
        st.latex(
            r"t_{\mathrm{seg}}=\min(1,\max(0,t_0)),"
            r"\qquad K=P+t_{\mathrm{seg}}(Q-P)"
        )
        st.latex(r"d_{\min}=\|C-K\|")

        st.write(
            "K es el punto del segmento más cercano al centro C. "
            "Si la distancia mínima supera el radio, el tramo está libre. "
            "El contacto con el borde también bloquea el paso."
        )

        if resultados:
            st.dataframe(
                pd.DataFrame(resultados),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("El escenario no tiene obstáculos.")

        st.caption(
            "Se utiliza una tolerancia de 0.00000001 m para clasificar "
            "los contactos con el borde. Si P y Q coinciden, t₀ no se "
            "calcula y K coincide con P."
        )
with simulacion:
    st.divider()
    st.subheader("Recorrido planificado")

    if not errores:
        # La firma permite retirar resultados anteriores cuando
        # cambian las posiciones, los obstáculos o el margen.
        firma = (
            float(ancho),
            float(alto),
            tuple(inicio),
            tuple(destino),
            float(margen),
            tuple(
                (
                    tuple(obstaculo["centro"]),
                    obstaculo["radio"],
                )
                for obstaculo in obstaculos
            ),
        )

        if st.session_state.get("firma_ruta") != firma:
            st.session_state.pop("ruta_calculada", None)
            st.session_state.pop("firma_ruta", None)

        extremos_con_margen = (
            segmento_libre(inicio, inicio, obstaculos, margen)
            and segmento_libre(destino, destino, obstaculos, margen)
        )

        if not extremos_con_margen:
            st.warning(
                "P y Q deben quedar fuera de los obstáculos "
                "y de sus márgenes de separación."
            )

        # Se limita el tamaño de esta primera implementación
        # porque comprueba cada pareja de puntos candidatos.
        demasiados_obstaculos = len(obstaculos) > 15

        if demasiados_obstaculos:
            st.warning(
                "Esta versión admite hasta 15 obstáculos "
                "para calcular el recorrido."
            )

        pulsado = st.button(
            "Calcular recorrido",
            type="primary",
            disabled=(
                not extremos_con_margen
                or demasiados_obstaculos
            ),
        )

        if pulsado:
            with st.spinner("Comprobando conexiones y buscando la ruta..."):
                ruta = calcular_ruta(
                    inicio,
                    destino,
                    obstaculos,
                    ancho,
                    alto,
                    margen,
                )

            st.session_state["ruta_calculada"] = ruta
            st.session_state["firma_ruta"] = firma

        resultado_actual = (
            st.session_state.get("firma_ruta") == firma
        )

        if resultado_actual:
            ruta = st.session_state["ruta_calculada"]

            if ruta is None:
                st.warning(
                    "No se encontró un recorrido con los puntos "
                    "candidatos generados y el margen seleccionado. "
                    "Esto no demuestra que no exista otra ruta."
                )
            else:
                figura_ruta, eje_ruta = plt.subplots(figsize=(9, 6))
                figura_ruta.patch.set_facecolor("white")
                eje_ruta.set_facecolor("white")

                for obstaculo in obstaculos:
                    centro = obstaculo["centro"]
                    radio = obstaculo["radio"]

                    eje_ruta.add_patch(
                        Circle(
                            centro,
                            radio + margen,
                            fill=False,
                            edgecolor="#94A3B8",
                            linestyle=":",
                            linewidth=1.2,
                        )
                    )

                    eje_ruta.add_patch(
                        Circle(
                            centro,
                            radio,
                            facecolor="#E2E8F0",
                            edgecolor="#64748B",
                            alpha=0.8,
                        )
                    )

                    eje_ruta.annotate(
                        obstaculo["nombre"],
                        centro,
                        ha="center",
                        va="center",
                    )

                eje_ruta.plot(
                    [inicio[0], destino[0]],
                    [inicio[1], destino[1]],
                    "--",
                    color="#94A3B8",
                    label="Conexión directa",
                )

                eje_ruta.plot(
                    ruta[:, 0],
                    ruta[:, 1],
                    "-o",
                    color="#2563EB",
                    linewidth=2,
                    markersize=3,
                    label="Recorrido planificado",
                )

                eje_ruta.scatter(
                    *inicio,
                    color="#15803D",
                    s=100,
                    label="Inicio P",
                    zorder=5,
                )

                eje_ruta.scatter(
                    *destino,
                    color="#7C3AED",
                    marker="*",
                    s=200,
                    label="Destino Q",
                    zorder=6,
                )

                eje_ruta.set_xlim(0, ancho)
                eje_ruta.set_ylim(0, alto)
                eje_ruta.set_aspect("equal", adjustable="box")
                eje_ruta.set_xlabel("x (m)")
                eje_ruta.set_ylabel("y (m)")
                eje_ruta.set_title("Recorrido libre de obstáculos")
                eje_ruta.grid(True, alpha=0.25)
                eje_ruta.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=2,
                    frameon=False,
                )

                st.pyplot(figura_ruta, use_container_width=True)
                plt.close(figura_ruta)

                st.caption(
                    "Los círculos punteados muestran el margen "
                    "de separación añadido a cada obstáculo."
                )

                longitudes = np.linalg.norm(
                    np.diff(ruta, axis=0),
                    axis=1,
                )
                longitud_total = float(np.sum(longitudes))
                distancia_recta = float(np.linalg.norm(destino - inicio))

                columna_1, columna_2, columna_3 = st.columns(3)

                columna_1.metric(
                    "Longitud del recorrido",
                    f"{longitud_total:.3f} m",
                )
                columna_2.metric(
                    "Distancia directa",
                    f"{distancia_recta:.3f} m",
                )
                columna_3.metric(
                    "Longitud adicional",
                    f"{max(0.0, longitud_total - distancia_recta):.3f} m",
                )

                st.success(
                    "Todos los segmentos del recorrido cumplen "
                    "la separación respecto a los obstáculos."
                )

                with calculos:
                    st.divider()
                    st.subheader("Tramos del recorrido planificado")

                    filas_tramos = []

                    for numero, (punto_a, punto_b) in enumerate(
                        zip(ruta[:-1], ruta[1:]),
                        1,
                    ):
                        vector_tramo = punto_b - punto_a

                        filas_tramos.append(
                            {
                                "Tramo": numero,
                                "Inicio x (m)": punto_a[0],
                                "Inicio y (m)": punto_a[1],
                                "Final x (m)": punto_b[0],
                                "Final y (m)": punto_b[1],
                                "Δx (m)": vector_tramo[0],
                                "Δy (m)": vector_tramo[1],
                                "Longitud (m)": float(
                                    np.linalg.norm(vector_tramo)
                                ),
                                "Cumple separación": segmento_libre(
                                    punto_a,
                                    punto_b,
                                    obstaculos,
                                    margen,
                                ),
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(filas_tramos),
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.caption(
                        "Se generan 32 puntos candidatos por obstáculo "
                        "y se utiliza Dijkstra para buscar el recorrido "
                        "más corto en el grafo de conexiones libres."
                    )
with simulacion:
    st.divider()
    st.subheader("Animación del dron")

    if errores:
        st.info("Corrige el escenario para poder reproducir la animación.")
    elif (
        st.session_state.get("firma_ruta") == firma
        and st.session_state.get("ruta_calculada") is not None
    ):
        st.caption(
            "La reproducción muestra el avance geométrico sobre la ruta. "
            "Su duración es ilustrativa y no representa un tiempo de vuelo físico."
        )

        if st.button("Preparar animación", key="preparar_animacion"):
            with st.spinner("Preparando los cuadros del recorrido..."):
                mostrar_animacion(
                    st.session_state["ruta_calculada"],
                    obstaculos,
                    ancho,
                    alto,
                    margen,
                )

            st.caption(
                "Utiliza los controles inferiores para reproducir, "
                "pausar o revisar los cuadros."
            )
    else:
        st.info(
            "Calcula primero un recorrido válido para habilitar la animación."
        )