"""
Visualizador de Analogías de Palabras
======================================

Aplicación web interactiva para explorar relaciones semánticas entre palabras
usando word embeddings (GloVe) y visualizarlas en un espacio tridimensional.

Autor: Juan Otero
"""

# ============================================================================
# IMPORTACIONES
# ============================================================================

import streamlit as st  # Framework para aplicaciones web
import gensim.downloader  # Para descargar modelos pre-entrenados de word embeddings
import numpy as np  # Operaciones numéricas con arrays
from sklearn.decomposition import PCA  # Reducción de dimensionalidad
from sklearn.preprocessing import StandardScaler  # Normalización de datos
import plotly.graph_objects as go  # Visualizaciones 3D interactivas

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Visualizador de Analogías de Palabras",
    page_icon="🔤",
    layout="wide"  # Aprovecha todo el ancho de la pantalla
)


# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

@st.cache_resource
def load_model():
    """
    Carga el modelo de word embeddings GloVe pre-entrenado.

    El decorador @st.cache_resource asegura que el modelo se cargue
    solo una vez y se mantenga en caché durante toda la sesión.

    Returns:
        gensim.models.KeyedVectors: Modelo GloVe con 400,000 palabras
                                     representadas en 50 dimensiones
    """
    return gensim.downloader.load("glove-wiki-gigaword-50")


def get_vectors_with_context(model, words, n_context=15):
    """
    Obtiene vectores de embeddings para las palabras base, calcula el
    vector derivado y añade palabras contextuales para mejorar la visualización.

    Args:
        model: Modelo de word embeddings cargado
        words (list): Lista de 4 palabras [palabra1, palabra2, palabra3, palabra4]
        n_context (int): Número de palabras contextuales a añadir

    Returns:
        tuple: (vectores, etiquetas, tipos, error)
            - vectores (np.array): Array de vectores (n_palabras x 50)
            - etiquetas (list): Nombres de las palabras
            - tipos (list): Tipo de cada palabra ('base', 'derived', 'context')
            - error (str): Mensaje de error si algo falla, None si todo OK

    Proceso:
        1. Obtiene vectores de las 4 palabras base del modelo
        2. Calcula palabra derivada: palabra1 - palabra2 + palabra3
        3. Busca palabras similares a cada palabra base para añadir contexto
        4. Retorna todos los vectores organizados por tipo
    """
    all_vectors = []
    all_labels = []
    all_types = []

    # --- PASO 1: Obtener vectores de las 4 palabras base ---
    for word in words:
        try:
            vector = model[word]  # Busca la palabra en el modelo
            all_vectors.append(vector)
            all_labels.append(word)
            all_types.append('base')
        except KeyError:
            # Si la palabra no existe en el vocabulario, retornar error
            return None, None, None, f"Palabra '{word}' no encontrada en el vocabulario."

    # --- PASO 2: Calcular palabra derivada usando aritmética vectorial ---
    # Fórmula: palabra1 - palabra2 + palabra3
    # Ejemplo: king - man + woman ≈ queen
    try:
        derived_vector = model[words[0]] - model[words[1]] + model[words[2]]
        all_vectors.append(derived_vector)
        derived_label = f"{words[0]}-{words[1]}+{words[2]}"
        all_labels.append(derived_label)
        all_types.append('derived')
    except KeyError as e:
        return None, None, None, f"Error al calcular derivada: {e}"

    # --- PASO 3: Añadir palabras contextuales ---
    # Las palabras contextuales ayudan a PCA a crear una mejor proyección 3D
    # porque proporcionan más puntos de referencia en el espacio semántico
    context_words = set()
    for word in words:
        try:
            # Busca las palabras más similares a cada palabra base
            similar = model.most_similar(word, topn=n_context//4 + 2)
            for sim_word, _ in similar:
                # Evita duplicados y palabras base
                if sim_word not in words and sim_word not in context_words:
                    context_words.add(sim_word)
                    if len(context_words) >= n_context:
                        break
        except:
            pass

    # Añadir vectores de palabras contextuales
    context_words = list(context_words)[:n_context]
    for word in context_words:
        try:
            all_vectors.append(model[word])
            all_labels.append(word)
            all_types.append('context')
        except:
            pass

    return np.array(all_vectors), all_labels, all_types, None


def reduce_dimensions(vectors):
    """
    Reduce la dimensionalidad de los vectores de 50D a 3D usando PCA.

    Args:
        vectors (np.array): Array de vectores de alta dimensión (n x 50)

    Returns:
        np.array: Vectores reducidos a 3 dimensiones (n x 3)

    Proceso:
        1. Normalización L2: Convierte todos los vectores a longitud unitaria
           Esto asegura que solo importe la dirección, no la magnitud
        2. Estandarización: Media = 0, Desviación estándar = 1
           Mejora la performance de PCA eliminando sesgos de escala
        3. PCA: Reduce de 50D a 3D preservando la máxima varianza posible
    """
    # PASO 1: Normalización L2 (unit vectors)
    # Cada vector se divide por su norma euclidiana
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # PASO 2: Estandarización (StandardScaler)
    # Centra los datos en 0 y los escala a varianza unitaria
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors_norm)

    # PASO 3: PCA - Análisis de Componentes Principales
    # Encuentra las 3 direcciones de máxima varianza en el espacio 50D
    # y proyecta los datos en ese subespacio 3D
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(vectors_scaled)

    return reduced


def plot_3d_enhanced(vectors_3d, labels, types, words, similarity=None, similar_words=None):
    """
    Crea una visualización 3D interactiva usando Plotly.

    Args:
        vectors_3d (np.array): Vectores reducidos a 3D
        labels (list): Etiquetas de cada punto
        types (list): Tipo de cada punto ('base', 'derived', 'context')
        words (list): Las 4 palabras originales de la analogía
        similarity (float): Similitud coseno entre derivada y palabra4
        similar_words (list): Top 10 palabras similares a la derivada

    Returns:
        plotly.graph_objects.Figure: Figura de Plotly lista para mostrar

    Elementos visuales:
        - Puntos grises (contexto): Palabras relacionadas de fondo
        - Puntos azules (base): Las 4 palabras de entrada
        - Punto rojo grande (derivada): Resultado de palabra1 - palabra2 + palabra3
        - Línea naranja: Conecta palabra1 con palabra2 (operación de resta)
        - Línea cyan: Conecta palabra3 con palabra4 (relación esperada)
        - Línea amarilla punteada: Conecta derivada con palabra4 (evaluación)
    """
    fig = go.Figure()

    # --- SEPARAR PUNTOS POR TIPO ---
    base_indices = [i for i, t in enumerate(types) if t == 'base']
    derived_indices = [i for i, t in enumerate(types) if t == 'derived']
    context_indices = [i for i, t in enumerate(types) if t == 'context']

    # --- TRAZA 1: Palabras de contexto (fondo, gris, semitransparente) ---
    if context_indices:
        fig.add_trace(go.Scatter3d(
            x=vectors_3d[context_indices, 0],
            y=vectors_3d[context_indices, 1],
            z=vectors_3d[context_indices, 2],
            mode='markers+text',
            marker=dict(
                size=5,
                color='rgb(120, 120, 120)',
                opacity=0.35,
                line=dict(color='white', width=0.5)
            ),
            text=[labels[i] for i in context_indices],
            textposition='top center',
            textfont=dict(size=7, color='lightgray'),
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
            name='Contexto',
            showlegend=True
        ))

    # --- TRAZA 2: Palabras base (azul, destacadas) ---
    fig.add_trace(go.Scatter3d(
        x=vectors_3d[base_indices, 0],
        y=vectors_3d[base_indices, 1],
        z=vectors_3d[base_indices, 2],
        mode='markers+text',
        marker=dict(
            size=16,
            color='rgb(100, 149, 237)',  # Azul cornflower
            opacity=1.0,
            line=dict(color='white', width=2)
        ),
        text=[labels[i] for i in base_indices],
        textposition='top center',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
        name='Palabras base',
        showlegend=True
    ))

    # --- TRAZA 3: Palabra derivada (rojo-naranja, más grande) ---
    fig.add_trace(go.Scatter3d(
        x=vectors_3d[derived_indices, 0],
        y=vectors_3d[derived_indices, 1],
        z=vectors_3d[derived_indices, 2],
        mode='markers+text',
        marker=dict(
            size=22,
            color='rgb(255, 69, 0)',  # Rojo-naranja brillante
            opacity=1.0,
            line=dict(color='white', width=2)
        ),
        text=[labels[i] for i in derived_indices],
        textposition='top center',
        textfont=dict(size=13, color='white'),
        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
        name='Derivada',
        showlegend=True
    ))

    # --- ENCONTRAR ÍNDICES DE PALABRAS ESPECÍFICAS ---
    idx_word1 = labels.index(words[0])
    idx_word2 = labels.index(words[1])
    idx_word3 = labels.index(words[2])
    idx_word4 = labels.index(words[3])
    idx_derived = next(i for i, t in enumerate(types) if t == 'derived')

    # --- LÍNEA 1: palabra1 ↔ palabra2 (naranja) ---
    # Representa la operación de resta en la fórmula
    fig.add_trace(go.Scatter3d(
        x=[vectors_3d[idx_word1, 0], vectors_3d[idx_word2, 0]],
        y=[vectors_3d[idx_word1, 1], vectors_3d[idx_word2, 1]],
        z=[vectors_3d[idx_word1, 2], vectors_3d[idx_word2, 2]],
        mode='lines',
        line=dict(color='orange', width=5),
        name=f'{words[0]} ↔ {words[1]}',
        showlegend=True
    ))

    # --- LÍNEA 2: palabra3 ↔ palabra4 (cyan) ---
    # Muestra la relación esperada (similar a la derivada)
    fig.add_trace(go.Scatter3d(
        x=[vectors_3d[idx_word3, 0], vectors_3d[idx_word4, 0]],
        y=[vectors_3d[idx_word3, 1], vectors_3d[idx_word4, 1]],
        z=[vectors_3d[idx_word3, 2], vectors_3d[idx_word4, 2]],
        mode='lines',
        line=dict(color='cyan', width=5),
        name=f'{words[2]} ↔ {words[3]}',
        showlegend=True
    ))

    # --- LÍNEA 3: derivada → palabra4 (amarilla punteada) ---
    # Evalúa qué tan cerca está la derivada de palabra4
    # Línea corta = analogía funciona bien
    fig.add_trace(go.Scatter3d(
        x=[vectors_3d[idx_derived, 0], vectors_3d[idx_word4, 0]],
        y=[vectors_3d[idx_derived, 1], vectors_3d[idx_word4, 1]],
        z=[vectors_3d[idx_derived, 2], vectors_3d[idx_word4, 2]],
        mode='lines',
        line=dict(color='yellow', width=7, dash='dash'),
        name=f'Derivada → {words[3]}',
        showlegend=True
    ))

    # --- TÍTULO CON INFORMACIÓN DE LA ANALOGÍA ---
    title_text = f'<b>{words[0]} - {words[1]} + {words[2]} ≈ {words[3]}</b>'
    if similarity is not None:
        title_text += f'<br><sub>Similitud (coseno) = {similarity:.4f}</sub>'

    # --- CONFIGURACIÓN DEL LAYOUT DEL GRÁFICO ---
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center', font=dict(size=15)),
        scene=dict(
            xaxis_title='PCA Componente 1',
            yaxis_title='PCA Componente 2',
            zaxis_title='PCA Componente 3',
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.4)),  # Ángulo de cámara
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', backgroundcolor='rgb(20,20,20)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', backgroundcolor='rgb(20,20,20)'),
            zaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', backgroundcolor='rgb(20,20,20)')
        ),
        template='plotly_dark',  # Tema oscuro
        showlegend=True,
        height=900,  # Altura del gráfico en píxeles
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.6)', font=dict(size=11))
    )

    return fig


# ============================================================================
# INTERFAZ DE USUARIO (STREAMLIT)
# ============================================================================

st.title("🔤 Visualizador de Analogías de Palabras")
st.markdown("""
Explora relaciones semánticas entre palabras usando **word embeddings** y visualízalas en 3D.

**Fórmula:** `palabra1 - palabra2 + palabra3 ≈ palabra4`
""")

# --- SIDEBAR (BARRA LATERAL) ---
with st.sidebar:
    st.header("⚙️ Configuración")

    # Ejemplos predefinidos de analogías
    st.markdown("### Ejemplos de analogías:")
    examples = {
        "Rey → Reina": ["king", "man", "woman", "queen"],
        "París → Madrid": ["paris", "france", "spain", "madrid"],
        "Tío → Tía": ["uncle", "man", "woman", "aunt"],
        "Caminando → Nadando": ["walking", "walked", "swimming", "swam"]
    }

    # Selector de ejemplos
    selected_example = st.selectbox(
        "Selecciona un ejemplo:",
        ["Personalizado"] + list(examples.keys())
    )

    st.markdown("---")

    # Control deslizante para número de palabras contextuales
    n_context = st.slider(
        "Palabras contextuales:",
        min_value=5,
        max_value=30,
        value=20,
        step=5,
        help="Más palabras = mejor visualización pero más lento"
    )

    st.markdown("---")

    # Leyenda de colores
    st.markdown("""
    **Leyenda:**
    - 🔵 Palabras base
    - 🔴 Palabra derivada
    - ⚪ Palabras contextuales
    - 🟠 Línea: palabra1 ↔ palabra2
    - 🔵 Línea: palabra3 ↔ palabra4
    - 🟡 Línea punteada: derivada → palabra4
    """)

# --- ÁREA PRINCIPAL: INPUTS DE PALABRAS ---
col1, col2, col3, col4 = st.columns(4)

# Determinar valores por defecto según ejemplo seleccionado
if selected_example != "Personalizado":
    default_words = examples[selected_example]
else:
    default_words = ["king", "man", "woman", "queen"]

# Campos de entrada para las 4 palabras
with col1:
    word1 = st.text_input("Palabra 1:", value=default_words[0], key="word1")
with col2:
    word2 = st.text_input("Palabra 2:", value=default_words[1], key="word2")
with col3:
    word3 = st.text_input("Palabra 3:", value=default_words[2], key="word3")
with col4:
    word4 = st.text_input("Palabra 4:", value=default_words[3], key="word4")

# Botón principal para generar visualización
generate_button = st.button("🚀 Generar Visualización", type="primary", use_container_width=True)

# --- PROCESAMIENTO CUANDO SE PRESIONA EL BOTÓN ---
if generate_button:
    # Normalizar entradas (minúsculas y sin espacios)
    words = [word1.strip().lower(), word2.strip().lower(),
             word3.strip().lower(), word4.strip().lower()]

    # Validar que todas las palabras estén completas
    if any(not w for w in words):
        st.error("⚠️ Por favor, completa las 4 palabras.")
    else:
        # PASO 1: Cargar modelo de word embeddings (con caché)
        model = load_model()

        # Barra de progreso para feedback visual
        progress_bar = st.progress(0)
        status_text = st.empty()

        # PASO 2: Obtener vectores y palabras contextuales
        status_text.text("Obteniendo vectores de palabras...")
        progress_bar.progress(20)

        vectors, labels, types, error = get_vectors_with_context(model, words, n_context)

        if error:
            st.error(f"❌ Error: {error}")
        else:
            # PASO 3: Encontrar palabras similares a la derivada
            status_text.text("Buscando palabras similares...")
            progress_bar.progress(40)

            derived_idx = next(i for i, t in enumerate(types) if t == 'derived')

            # Usa similitud coseno para encontrar las 10 palabras más cercanas
            # similar_by_vector compara el vector derivado con todo el vocabulario
            similar_words = model.similar_by_vector(vectors[derived_idx], topn=10)

            # PASO 4: Calcular similitud coseno entre derivada y palabra4
            word4_idx = labels.index(words[3])
            derived_vec = vectors[derived_idx]
            word4_vec = vectors[word4_idx]

            # Fórmula de similitud coseno: cos(θ) = (A·B) / (||A|| × ||B||)
            # Resultado entre -1 y 1, donde 1 = vectores idénticos
            similarity = np.dot(derived_vec, word4_vec) / (
                np.linalg.norm(derived_vec) * np.linalg.norm(word4_vec)
            )

            # PASO 5: Reducir dimensionalidad de 50D a 3D con PCA
            status_text.text("Aplicando PCA...")
            progress_bar.progress(60)

            vectors_3d = reduce_dimensions(vectors)

            # PASO 6: Crear visualización 3D
            status_text.text("Generando visualización 3D...")
            progress_bar.progress(80)

            fig = plot_3d_enhanced(vectors_3d, labels, types, words, similarity, similar_words)

            progress_bar.progress(100)
            status_text.text("✅ Completado!")

            # --- MOSTRAR RESULTADOS ---
            st.markdown("---")

            # MÉTRICAS EN 3 COLUMNAS
            col1, col2, col3 = st.columns(3)

            # Métrica 1: Similitud coseno
            with col1:
                st.metric(
                    "Similitud Coseno",
                    f"{similarity:.4f}",
                    help="Similitud entre palabra derivada y palabra4 (1.0 = idénticas)"
                )

            # Métrica 2: Ranking de palabra4 en top 10
            with col2:
                top_5_words = [w for w, _ in similar_words[:5]]
                rank = next((i+1 for i, (w, _) in enumerate(similar_words) if w == words[3]), None)

                if rank and rank <= 5:
                    st.metric(
                        "Ranking de palabra4",
                        f"#{rank} en Top 10",
                        delta="✓ Excelente",
                        delta_color="normal"
                    )
                elif rank:
                    st.metric(
                        "Ranking de palabra4",
                        f"#{rank} en Top 10",
                        delta="Aceptable",
                        delta_color="off"
                    )
                else:
                    st.metric(
                        "Ranking de palabra4",
                        "Fuera del Top 10",
                        delta="Revisar analogía",
                        delta_color="inverse"
                    )

            # Métrica 3: Total de vectores
            with col3:
                st.metric(
                    "Vectores totales",
                    len(vectors),
                    help=f"4 palabras base + 1 derivada + {len([t for t in types if t == 'context'])} contextuales"
                )

            # GRÁFICO 3D INTERACTIVO
            st.plotly_chart(fig, use_container_width=True)

            # TOP 10 PALABRAS SIMILARES (en grid de 5 columnas)
            st.markdown("### 🔍 Top 10 palabras más cercanas a la derivada")

            cols = st.columns(5)
            for i, (word, score) in enumerate(similar_words[:10]):
                with cols[i % 5]:
                    # Destacar palabra4 si aparece en el top 10
                    if word == words[3]:
                        st.success(f"**★ {i+1}. {word}**\n\n`{score:.4f}`")
                    else:
                        st.info(f"**{i+1}. {word}**\n\n`{score:.4f}`")

            # Limpiar indicadores de progreso
            progress_bar.empty()
            status_text.empty()

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Juan Otero | Visualización de embeddings
</div>
""", unsafe_allow_html=True)
