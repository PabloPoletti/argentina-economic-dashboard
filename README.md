# 📊 Argentina Economic Dashboard

**🚀 [Ver Dashboard en Vivo](https://argentina-economic-dashboard.streamlit.app/)**

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF6B6B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**SeriesEcon Dashboard** es una aplicación web moderna e interactiva para el análisis avanzado de series económicas de Argentina. Utiliza datos en tiempo real de la API oficial del gobierno argentino y proporciona herramientas estadísticas sofisticadas para economistas, analistas e investigadores.

## 🌟 Características Principales

### 📈 **Exploración de Series Temporales**
- **Interfaz intuitiva** con filtros avanzados por ámbito geográfico, categoría económica y organismo
- **Visualizaciones interactivas** con Plotly: gráficos de líneas, áreas y barras
- **Normalización de datos** con opción de base 100
- **Métricas en tiempo real** de completitud y cobertura temporal

### 🔬 **Análisis Estadístico Avanzado**
- **Estadísticas descriptivas completas**: media, mediana, desviación estándar, asimetría, curtosis
- **Descomposición estacional** automática de series temporales
- **Pronósticos** utilizando modelos Exponential Smoothing y ARIMA
- **Análisis de distribución** con histogramas interactivos
- **Intervalos de confianza** para proyecciones

### 🔗 **Análisis de Correlaciones**
- **Matriz de correlación interactiva** con heatmaps
- **Identificación automática** de correlaciones destacadas
- **Análisis de Componentes Principales (PCA)** para reducción de dimensionalidad
- **Gráficos de varianza explicada**

### 📊 **Dashboard Ejecutivo**
- **Vista panorámica** del ecosistema de datos económicos
- **Métricas agregadas** por categoría y organismo
- **Distribución geográfica** de series disponibles
- **Indicadores de actualización** y completitud

### 💾 **Exportación Avanzada**
- **Múltiples formatos**: CSV, Excel con múltiples hojas
- **Metadatos incluidos** en exportaciones
- **Descarga de gráficos** en alta resolución

## 🚀 Instalación y Configuración

### Prerrequisitos
```bash
Python 3.8 o superior
pip (gestor de paquetes de Python)
```

### Instalación
1. **Clona el repositorio**:
```bash
git clone https://github.com/tu-usuario/seriesecon.git
cd seriesecon
```

2. **Instala las dependencias**:
```bash
pip install -r requirements.txt
```

3. **Ejecuta la aplicación**:
```bash
streamlit run streamlit_app.py
```

4. **Abre tu navegador** en `http://localhost:8501`

### 🐳 Docker (Opcional)
```bash
# Construir la imagen
docker build -t seriesecon .

# Ejecutar el contenedor
docker run -p 8501:8501 seriesecon
```

## 📚 Guía de Uso

### 🎯 Comenzando
1. **Selecciona filtros** en el panel lateral izquierdo
2. **Elige series** de interés usando los filtros de categoría económica
3. **Configura parámetros** como escala (nominal/real) y rango de fechas
4. **Explora** las diferentes páginas del dashboard

### 🔍 Navegación por Páginas

#### 🏠 **Explorar Series**
- Visualiza series seleccionadas con gráficos interactivos
- Cambia entre tipos de vista (gráfico, tabla, estadísticas)
- Normaliza datos para comparación
- Descarga datos en múltiples formatos

#### 📈 **Análisis Avanzado**
- Selecciona una serie para análisis detallado
- Observa descomposición estacional automática
- Genera pronósticos con diferentes métodos
- Analiza distribución de datos

#### 🔍 **Correlaciones**
- Compara múltiples series simultáneamente
- Identifica relaciones entre variables económicas
- Realiza análisis de componentes principales
- Encuentra patrones ocultos en los datos

#### 📊 **Dashboard Ejecutivo**
- Obtén una vista panorámica del sistema
- Explora distribución de datos por categoría
- Identifica organismos más activos
- Monitorea actualización de datos

### 🎛️ Filtros Disponibles

| Filtro | Descripción |
|--------|-------------|
| **Ámbito Geográfico** | Nacional, provincial o municipal |
| **Categoría Económica** | Inflación, PIB, Empleo, Comercio Exterior, etc. |
| **Tema** | Categorización temática oficial |
| **Organismo** | INDEC, BCRA, Ministerios, etc. |
| **Frecuencia** | Diaria, mensual, trimestral, anual |
| **Rango de Fechas** | Período específico de análisis |

## 🔧 Tecnologías Utilizadas

### Backend y Análisis
- **[Streamlit](https://streamlit.io/)** - Framework de aplicaciones web
- **[Pandas](https://pandas.pydata.org/)** - Manipulación de datos
- **[NumPy](https://numpy.org/)** - Computación numérica
- **[SciPy](https://scipy.org/)** - Análisis estadístico
- **[Scikit-learn](https://scikit-learn.org/)** - Machine learning
- **[Statsmodels](https://www.statsmodels.org/)** - Modelos estadísticos y econométricos

### Visualización
- **[Plotly](https://plotly.com/python/)** - Gráficos interactivos
- **[Seaborn](https://seaborn.pydata.org/)** - Visualizaciones estadísticas
- **[Matplotlib](https://matplotlib.org/)** - Gráficos base

### UI/UX
- **[Streamlit Option Menu](https://github.com/victoryhb/streamlit-option-menu)** - Navegación moderna
- **[Streamlit AgGrid](https://github.com/PabloFonseca/streamlit-aggrid)** - Tablas interactivas

## 📊 Fuente de Datos

Los datos provienen de la **API oficial del Gobierno de Argentina**:
- **Base URL**: `https://apis.datos.gob.ar/series/api/`
- **Metadatos**: Series económicas oficiales de múltiples organismos
- **Actualización**: Automática según calendario oficial
- **Cobertura**: Nacional, provincial y municipal

### Organismos Incluidos
- **INDEC** - Instituto Nacional de Estadística y Censos
- **BCRA** - Banco Central de la República Argentina
- **Ministerio de Economía**
- **ANSES** - Administración Nacional de la Seguridad Social
- **Y muchos más...**

## 🔬 Metodologías de Análisis

### Análisis de Series Temporales
- **Descomposición Estacional**: Separación de tendencia, estacionalidad y ruido
- **Suavizado Exponencial**: Modelos de Holt-Winters para pronósticos
- **ARIMA**: Modelos autoregresivos integrados de media móvil
- **Análisis de Estacionariedad**: Pruebas de raíz unitaria

### Análisis Multivariado
- **Matriz de Correlación**: Análisis de dependencias lineales
- **PCA**: Reducción de dimensionalidad y análisis de factores
- **Clustering**: Agrupación de series similares
- **Análisis de Componentes**: Identificación de factores subyacentes

### Estadísticas Descriptivas
- **Medidas de Tendencia Central**: Media, mediana, moda
- **Medidas de Dispersión**: Desviación estándar, coeficiente de variación
- **Medidas de Forma**: Asimetría y curtosis
- **Análisis de Outliers**: Detección de valores atípicos

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

### 🐛 Reportar Bugs
- Usa el sistema de **Issues** de GitHub
- Incluye **pasos para reproducir** el problema
- Adjunta **screenshots** si es relevante

### 💡 Solicitar Features
- Describe claramente la **funcionalidad deseada**
- Explica el **caso de uso**
- Considera la **viabilidad técnica**

## 📝 Roadmap

### Versión 2.0 (Próximamente)
- [ ] **Análisis de cointegración** entre series
- [ ] **Modelos VAR** (Vector Autoregression)
- [ ] **Análisis de causalidad de Granger**
- [ ] **Detección automática de cambios estructurales**
- [ ] **API REST** para acceso programático
- [ ] **Alertas automáticas** por email/Slack

### Versión 2.1
- [ ] **Análisis de volatilidad** (GARCH)
- [ ] **Machine Learning** para pronósticos
- [ ] **Análisis de sentimiento** de noticias económicas
- [ ] **Integración con APIs internacionales**

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Autor

- **Pablo Poletti** - *Desarrollo inicial* - [@PabloPoletti](https://github.com/PabloPoletti)

## 🔗 Proyectos Relacionados

- **[SeriesEcon Original](https://github.com/PabloPoletti/seriesecon)** - Versión original del dashboard
- **[Precios Argentina](https://github.com/PabloPoletti/Precios1)** - Dashboard de análisis de precios
- **[Esperanza de Vida](https://github.com/PabloPoletti/esperanza-vida-2)** - Análisis demográfico y esperanza de vida

## 🙏 Agradecimientos

- **Gobierno de Argentina** por proporcionar datos abiertos
- **INDEC y BCRA** por la calidad de las series económicas
- **Comunidad de Streamlit** por el framework excepcional
- **Comunidad de Python** por las librerías de análisis de datos

## 📞 Contacto

- **Email**: lic.poletti@gmail.com
- **LinkedIn**: [Pablo Poletti](https://www.linkedin.com/in/pablom-poletti/)
- **GitHub**: [@PabloPoletti](https://github.com/PabloPoletti)

---

<div align="center">

**⭐ Si este proyecto te resulta útil, ¡considera darle una estrella en GitHub! ⭐**

[🐛 Reportar Bug](https://github.com/PabloPoletti/argentina-economic-dashboard/issues/new?labels=bug) | [💡 Solicitar Feature](https://github.com/PabloPoletti/argentina-economic-dashboard/issues/new?labels=enhancement) | [📖 Documentación](https://github.com/PabloPoletti/argentina-economic-dashboard/wiki)

</div>
