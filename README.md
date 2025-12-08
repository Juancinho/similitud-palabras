# 🔤 Visualizador de Analogías de Palabras

<div align="center"

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

Una aplicación web interactiva para explorar y visualizar relaciones semánticas entre palabras usando **word embeddings** (GloVe) en un espacio tridimensional. Hecho por Juan Otero, para la asignatura de Historia de las Matemáticas.

[🚀 Demo en Vivo](https://juancinho-similitud-palabras-app-streamlit-io5gir.streamlit.app/)

</div>

---

## Tabla de Contenidos

- [Características](#características)
- [Instalación](#instalación)
- [Uso](#uso)
- [Arquitectura Técnica](#arquitectura-técnica)
- [Cómo Funciona](#cómo-funciona)
- [Ejemplos](#ejemplos)
- [Licencia](#licencia)
- [Autor](#autor)

---

## Características.

### Funcionalidades Principales.

- **Visualización 3D Interactiva**: Explora analogías de palabras en un espacio tridimensional rotable y zoomable.
- **Aritmética Vectorial**: Calcula `palabra1 - palabra2 + palabra3 ≈ palabra4`
- **Word Embeddings Pre-entrenados**: Utiliza GloVe (400,000 palabras, 50 dimensiones).
- **Reducción de Dimensionalidad**: PCA optimizado con normalización L2 y estandarización.
- **Palabras Contextuales**: Añade automáticamente palabras relacionadas para mejor visualización.
- **Análisis de Similitud**: Calcula similitud coseno y ranking de palabras.

## Instalación.

### Requisitos.

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación rápida.

```bash
# 1. Clonar el repositorio
git clone https://github.com/Juancinho/similitud-palabras.git
cd word-analogy-visualizer

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación
streamlit run app_streamlit.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Dependencias

| Paquete        | Versión | Propósito             |
| -------------- | ------- | --------------------- |
| `streamlit`    | ≥1.28.0 | Framework web         |
| `gensim`       | ≥4.3.0  | Word embeddings       |
| `numpy`        | ≥1.24.0 | Operaciones numéricas |
| `scikit-learn` | ≥1.3.0  | PCA y preprocessing   |
| `plotly`       | ≥5.18.0 | Visualización 3D      |

---

## 🚀 Uso.

![Uso1](https://github.com/Juancinho/similitud-palabras/blob/main/img/1.png)
![Uso2](https://github.com/Juancinho/similitud-palabras/blob/main/img/2.png)

1. **Abrir la aplicación**
   
   ```bash
   streamlit run app_streamlit.py
   ```

2. **Seleccionar ejemplo o ingresar palabras personalizadas**
   
   - Usa el selector en la sidebar para ejemplos predefinidos.
   - O ingresa 4 palabras manualmente.

3. **Ajustar configuración** (opcional)
   
   - Palabras contextuales: 5-30 (recomendado: 20).

4. **Generar visualización**
   
   - Click en "🚀 Generar Visualización".
   - Espera 5-10 segundos la primera vez (carga del modelo).

5. **Explorar resultados**
   
   - Rota el gráfico 3D con el mouse.
   - Revisa métricas de similitud.
   - Analiza top 10 palabras similares.

## Arquitectura técnica.

![arquitectura](https://github.com/Juancinho/similitud-palabras/blob/main/img/arquitectura.png)

### Algoritmos clave.

#### 1. Aritmética vectorial.

```python
# Fórmula de analogía
derivada = palabra1 - palabra2 + palabra3

# Ejemplo:
# king - man + woman ≈ queen
# [0.2, 0.5, ...] - [0.1, 0.3, ...] + [0.15, 0.4, ...] = [0.25, 0.6, ...]
```

#### 2. Similitud coseno.

```python
# Mide el ángulo entre dos vectores
similitud = (A · B) / (||A|| × ||B||)

# Rango: [-1, 1]
# 1.0  = Vectores idénticos
# 0.0  = Vectores perpendiculares
# -1.0 = Vectores opuestos
```

#### 3. PCA (Análisis de Componentes Principales).

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Normalizar vectores 
vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# 2. Centrar y Escalar
scaler = StandardScaler()
vectors_scaled = scaler.fit_transform(vectors_norm)

# 3. Aplicar PCA
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(vectors_scaled)
```

El uso de [PCA](https://es.wikipedia.org/wiki/An%C3%A1lisis_de_componentes_principales) (Principal Component Analysis) es fundamental para visualizar espacios de alta dimensión. A continuación se detalla el proceso matemático aplicado en esta herramienta:

**1. El Problema de la dimensionalidad.**
Los vectores de GloVe existen en $\mathbb{R}^{50}$. Visualizar 50 ejes ortogonales es imposible para la percepción humana. Necesitamos proyectar estos datos a $\mathbb{R}^{3}$ minimizando la pérdida de información estructural.

**2. Preprocesamiento crítico.**

* **Normalización euclídea:**
  
  En word embeddings, la magnitud del vector a menudo correlaciona con la frecuencia de la palabra en el corpus de entrenamiento, mientras que la información semántica reside principalmente en la *dirección* del vector. Sin normalizar, palabras con magnitudes grandes dominarían la varianza explicada por el PCA, distorsionando la visualización de similitudes semánticas.
  
  Para ello, dividimos cada vector $v$ por su norma euclídea: $v_{norm} = \frac{v}{||v||_2}$. Esto proyecta todos los puntos sobre la superficie de una hiperesfera unitaria.

* **Centrado de datos (estandarización):**
  
  PCA es una técnica basada en la varianza que rota los ejes. Para que esta rotación encuentre las direcciones de máxima varianza correctamente alrededor del conjunto de datos actual (el subconjunto de palabras seleccionado), el origen del sistema de coordenadas debe coincidir con el centroide de los datos (media cero).
  
  Restamos la media $\mu$ de cada dimensión: $x_{centrado} = x - \mu$. Además, escalamos a varianza unitaria para que ninguna dimensión domine sobre otras artificialmente.

**3. Reducción de Dimensionalidad**
PCA busca una transformación ortogonal tal que los primeros ejes (Componentes Principales) retengan la mayor cantidad de "información" (varianza) posible. La aplicamos usando la librería de Python `sklearn`.

---

## Cómo Funciona.

### Word Embeddings (GloVe).

Los **word embeddings** son representaciones vectoriales de palabras donde:

- Cada palabra es un vector de 50 números.
- Palabras similares tienen vectores cercanos.
- Relaciones semánticas se preservan como direcciones vectoriales.

**Ejemplo:**

```
king   = [0.23, -0.15, 0.67, ..., 0.34]  (50 valores)
queen  = [0.25, -0.13, 0.68, ..., 0.36]  (50 valores)
man    = [0.12, -0.08, 0.45, ..., 0.23]  (50 valores)
woman  = [0.14, -0.06, 0.46, ..., 0.25]  (50 valores)
```

### Analogías vectoriales.

Las analogías funcionan por **paralelismo vectorial**:

```
king - man ≈ queen - woman

Por lo tanto:
king - man + woman ≈ queen
```

El vector `king - man` es paralelo a `queen - woman`, representando el concepto de "realeza" independiente del género.

---

## Ejemplos.

### 1. Género.

```python
Input:  king, man, woman, queen
Output: Similitud = 0.8234
Top 1:  queen (0.8234)
```

### 2. Geografía.

```python
Input:  paris, france, spain, madrid
Output: Similitud = 0.7456
Top 1:  madrid (0.7456)
```

### 3. Familia.

```python
Input:  uncle, man, woman, aunt
Output: Similitud = 0.7892
Top 1:  aunt (0.7892)
```

### 4. Verbos Conjugados.

```python
Input:  walking, walked, swimming, swam
Output: Similitud = 0.6543
Top 3:  swam (0.6543)
```

### 5. Capital de País.

```python
Input:  tokyo, japan, france, paris
Output: Similitud = 0.7123
Top 1:  paris (0.7123)
```

---

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

```
MIT License

Copyright (c) 2025 Juan Otero

Permission is hereby granted, free of charge...
```

---

## 👨‍💻 Autor

**Juan Otero**

- GitHub: [@Juancinho](https://github.com/Juancinho)
- Linkedin: [Juan Otero Rivas](https://www.linkedin.com/in/juan-otero-rivas-4568471b2/)

Hecho con 🎧 y mucho ☕por Juan Otero
