# 🔤 Visualizador de Analogías de Palabras

<div align="center"

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

Una aplicación web interactiva para explorar y visualizar relaciones semánticas entre palabras usando **word embeddings** (GloVe) en un espacio tridimensional.

[🚀 Demo en Vivo](https://juancinho-similitud-palabras-app-streamlit-io5gir.streamlit.app/) | [📖 Documentación](#características) | [🐛 Reportar Bug](https://github.com/Juancinho/similitud-palabras/issues)

</div>

---

## 📋 Tabla de Contenidos

- [Características](#características)
- [Demo](#demo)
- [Instalación](#instalación)
- [Uso](#uso)
- [Arquitectura Técnica](#arquitectura-técnica)
- [Cómo Funciona](#cómo-funciona)
- [Ejemplos](#ejemplos)
- [Contribuir](#contribuir)
- [Licencia](#licencia)
- [Autor](#autor)

---

## ✨ Características

### 🎯 Funcionalidades Principales

- **Visualización 3D Interactiva**: Explora analogías de palabras en un espacio tridimensional rotable y zoomable
- **Aritmética Vectorial**: Calcula `palabra1 - palabra2 + palabra3 ≈ palabra4`
- **Word Embeddings Pre-entrenados**: Utiliza GloVe (400,000 palabras, 50 dimensiones)
- **Reducción de Dimensionalidad**: PCA optimizado con normalización L2 y estandarización
- **Palabras Contextuales**: Añade automáticamente palabras relacionadas para mejor visualización
- **Análisis de Similitud**: Calcula similitud coseno y ranking de palabras
- **Ejemplos Predefinidos**: Rey→Reina, París→Madrid, Tío→Tía, y más
- **Interfaz Responsive**: Funciona en desktop, tablet y móvil

### 🛠️ Características Técnicas

- **Caché Inteligente**: El modelo se carga solo una vez y se mantiene en memoria
- **Feedback Visual**: Barra de progreso en tiempo real durante el procesamiento
- **Manejo de Errores**: Validación de palabras y mensajes de error descriptivos
- **Optimización**: Procesamiento eficiente de 400K palabras en segundos
- **Escalable**: Arquitectura modular fácil de extender

## 📦 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 2 GB de RAM mínimo (para cargar el modelo)
- Conexión a internet (primera carga del modelo)

### Instalación Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/Juancinho/similitud-palabras.git
cd word-analogy-visualizer

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
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

## 🚀 Uso

![Uso1](https://github.com/Juancinho/similitud-palabras/blob/main/img/1.png)
![Uso2](https://github.com/Juancinho/similitud-palabras/blob/main/img/2.png)

### Uso Básico

1. **Abrir la aplicación**
   
   ```bash
   streamlit run app_streamlit.py
   ```

2. **Seleccionar ejemplo o ingresar palabras personalizadas**
   
   - Usa el selector en la sidebar para ejemplos predefinidos
   - O ingresa 4 palabras manualmente

3. **Ajustar configuración** (opcional)
   
   - Palabras contextuales: 5-30 (recomendado: 20)

4. **Generar visualización**
   
   - Click en "🚀 Generar Visualización"
   - Espera 5-10 segundos la primera vez (carga del modelo)

5. **Explorar resultados**
   
   - Rota el gráfico 3D con el mouse
   - Revisa métricas de similitud
   - Analiza top 10 palabras similares

### Uso desde Línea de Comandos

```bash
# Versión CLI (sin interfaz web)
python main.py
```

---

## 🏗️ Arquitectura Técnica

### Pipeline de Procesamiento

```
┌─────────────────┐
│   Entrada de    │
│   4 Palabras    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Cargar Modelo  │
│   GloVe 50D     │ ──► Caché (1ª vez: 5-10s, después: instantáneo)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Obtener Vectores│
│   + Contexto    │ ──► 4 palabras + 1 derivada + N contextuales
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Aritmética      │
│  Vectorial      │ ──► palabra1 - palabra2 + palabra3
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Normalización  │
│   L2 + Scale    │ ──► Preparación para PCA
└────────┬────────┘
         │
         v
┌─────────────────┐
│  PCA (50D→3D)   │ ──► Reducción de dimensionalidad
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Visualización  │
│   Plotly 3D     │ ──► Gráfico interactivo
└─────────────────┘
```

### Algoritmos Clave

#### 1. Aritmética Vectorial

```python
# Fórmula de analogía
derivada = palabra1 - palabra2 + palabra3

# Ejemplo:
# king - man + woman ≈ queen
# [0.2, 0.5, ...] - [0.1, 0.3, ...] + [0.15, 0.4, ...] = [0.25, 0.6, ...]
```

#### 2. Similitud Coseno

```python
# Mide el ángulo entre dos vectores
similitud = (A · B) / (||A|| × ||B||)

# Rango: [-1, 1]
# 1.0  = Vectores idénticos
# 0.0  = Vectores perpendiculares
# -1.0 = Vectores opuestos
```

#### 3. PCA (Análisis de Componentes Principales)

```python
# Pasos de reducción:
1. Normalización L2    → Vectores unitarios
2. Estandarización    → Media=0, Std=1
3. PCA                → Encuentra 3 direcciones de máxima varianza
4. Proyección         → Mapea datos a espacio 3D
```

---

## 🔬 Cómo Funciona

### Word Embeddings (GloVe)

Los **word embeddings** son representaciones vectoriales de palabras donde:

- Cada palabra es un vector de 50 números
- Palabras similares tienen vectores cercanos
- Relaciones semánticas se preservan como direcciones vectoriales

**Ejemplo:**

```
king   = [0.23, -0.15, 0.67, ..., 0.34]  (50 valores)
queen  = [0.25, -0.13, 0.68, ..., 0.36]  (50 valores)
man    = [0.12, -0.08, 0.45, ..., 0.23]  (50 valores)
woman  = [0.14, -0.06, 0.46, ..., 0.25]  (50 valores)
```

### Analogías Vectoriales

Las analogías funcionan por **paralelismo vectorial**:

```
king - man ≈ queen - woman

Por lo tanto:
king - man + woman ≈ queen
```

Visualmente en 2D (simplificado):

```
     queen •
            ↗
           /
     king •    woman •
       ↘   ↗
         ×
       ↙   ↖
    man •
```

El vector `king - man` es paralelo a `queen - woman`, representando el concepto de "realeza" independiente del género.

### Reducción de Dimensionalidad

**Problema:** No podemos visualizar 50 dimensiones
**Solución:** PCA reduce a 3D preservando relaciones

**Proceso:**

1. **Normalización L2**: Todos los vectores a longitud 1
2. **Estandarización**: Centra datos en origen
3. **PCA**: Encuentra 3 ejes principales de varianza
4. **Proyección**: Mapea puntos al nuevo espacio 3D

**Trade-off:**

- ✅ Podemos visualizar
- ⚠️ Perdemos ~20-30% de información
- ✅ Relaciones principales se preservan

---

## 📚 Ejemplos

### 1. Género

```python
Input:  king, man, woman, queen
Output: Similitud = 0.8234 ⭐
Top 1:  queen (0.8234)
```

### 2. Geografía

```python
Input:  paris, france, spain, madrid
Output: Similitud = 0.7456 ⭐
Top 1:  madrid (0.7456)
```

### 3. Familia

```python
Input:  uncle, man, woman, aunt
Output: Similitud = 0.7892 ⭐
Top 1:  aunt (0.7892)
```

### 4. Verbos Conjugados

```python
Input:  walking, walked, swimming, swam
Output: Similitud = 0.6543 ✓
Top 3:  swam (0.6543)
```

### 5. Capital de País

```python
Input:  tokyo, japan, france, paris
Output: Similitud = 0.7123 ⭐
Top 1:  paris (0.7123)
```

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Aquí hay algunas formas de contribuir:

### Reportar Bugs

1. Abre un [Issue](https://github.com/Juancinho/similitud-palabras/issues)
2. Describe el bug claramente
3. Incluye pasos para reproducirlo
4. Añade screenshots si es posible

### Proponer Features

1. Abre un [Issue](https://github.com/Juancinho/similitud-palabras/issues) con etiqueta "enhancement"
2. Describe la funcionalidad deseada
3. Explica por qué sería útil

### Submit Pull Request

1. Fork el repositorio
2. Crea tu branch (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Ideas para Contribuir

- [ ] Añadir más modelos de embeddings (Word2Vec, FastText, BERT)
- [ ] Implementar t-SNE como alternativa a PCA
- [ ] Exportar visualizaciones como imagen/video
- [ ] Modo oscuro/claro toggle
- [ ] Soporte para múltiples idiomas
- [ ] API REST para uso programático
- [ ] Tests unitarios con pytest
- [ ] Dockerfile para containerización

---

## 📊 Performance

| Operación         | Primera Vez | Subsecuentes        |
| ----------------- | ----------- | ------------------- |
| Carga de modelo   | ~8 segundos | Instantáneo (caché) |
| Obtener vectores  | ~0.5 seg    | ~0.5 seg            |
| PCA (5 palabras)  | ~0.1 seg    | ~0.1 seg            |
| PCA (25 palabras) | ~0.3 seg    | ~0.3 seg            |
| Renderizado 3D    | ~1 seg      | ~1 seg              |
| **Total**         | **~10 seg** | **~2 seg**          |

---

## 🔒 Privacidad y Seguridad

- ✅ No se recopilan datos del usuario
- ✅ Todo el procesamiento es local
- ✅ Sin cookies ni tracking
- ✅ Open source (código auditable)
- ✅ Sin dependencias sospechosas

---

## 📄 Licencia

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

---

## 🙏 Agradecimientos

- **GloVe Team** - Por los word embeddings pre-entrenados
- **Streamlit** - Por el framework web increíble
- **Plotly** - Por las visualizaciones 3D interactivas
- **Gensim** - Por la biblioteca de NLP

---

## 📚 Referencias

### Papers

1. Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"
   
   - [Paper](https://arxiv.org/abs/1301.3781)

2. Pennington et al. (2014) - "GloVe: Global Vectors for Word Representation"
   
   - [Paper](https://nlp.stanford.edu/pubs/glove.pdf)

3. Levy & Goldberg (2014) - "Linguistic Regularities in Sparse and Explicit Word Representations"
   
   - [Paper](https://www.aclweb.org/anthology/W14-1618/)

###](https://jalammar.github.io/illustrated-word2vec/)

- [GloVe Homepage](https://nlp.stanford.edu/projects/glove/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Word Embeddings Explained](https://jalammar.github.io/illustrated-word2vec/)

---

## ⭐ Star History

Si este proyecto te fue útil, ¡deja una estrella! ⭐

---
