import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import mimetypes
from datetime import datetime
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Exploración datos FIUT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar los datos
@st.cache_data
def cargar_datos(ruta='estructura_archivos.csv'):
    """Carga los datos del archivo CSV o genera un DataFrame vacío si no existe"""
    try:
        df = pd.read_csv(ruta)
        return df
    except FileNotFoundError:
        st.error(f"Archivo {ruta} no encontrado. Por favor ejecuta primero el script de generación.")
        return pd.DataFrame()

# Función para procesar y limpiar los datos
def procesar_datos(df):
    """Procesa y limpia los datos para el análisis"""
    if df.empty:
        return df
    
    # Filtrar solo archivos (no directorios)
    df = df[df['tipo'] == 'Archivo'].copy()
    
    # Eliminar filas con extensión vacía
    df = df[df['extension'] != ''].copy()
    
    # Eliminar archivos .ipynb
    df = df[df['extension'] != '.ipynb'].copy()
    
    # Extraer dimensión de la ruta
    dims = []
    for ruta in df['ruta_relativa']:
        dim_encontrada = False
        for i in range(1, 8):
            dim_str = f"Dimensión {i}"
            if dim_str in ruta:
                dims.append(dim_str)
                dim_encontrada = True
                break
        if not dim_encontrada:
            dims.append('Sin clasificación')
    
    df['dimensiones'] = dims
    
    # Verificar columnas institucional/territorial
    if 'institucional' not in df.columns or 'territorial' not in df.columns:
        inst = []
        terr = []
        for ruta in df['ruta_relativa']:
            partes = ruta.split('\\')
            inst.append(partes[0] == 'Institucional')
            terr.append(partes[0] == 'Territorial')
        
        df['institucional'] = inst
        df['territorial'] = terr
    
    return df.reset_index(drop=True)

# Función para crear gráfico de barras institucional vs territorial
def crear_grafico_institucional_territorial(df):
    conteo = {
        'Institucional': df['institucional'].sum(),
        'Territorial': df['territorial'].sum()
    }
    
    fig = go.Figure([
        go.Bar(
            x=list(conteo.keys()),
            y=list(conteo.values()),
            marker_color=['#0A5C99', '#FEC109'],  # Cambiado a azul oscuro y amarillo
            text=list(conteo.values()),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Distribución de Archivos por Categoría',
        yaxis_title='Número de Archivos',
        template='plotly_white',
        height=400
    )
    
    return fig

# Función para crear gráfico de distribución de extensiones
def crear_grafico_extensiones(df, filtro=None):
    # Aplicar filtro si es necesario
    if filtro == 'institucional':
        df_temp = df[df['institucional'] == True]
        titulo = 'Distribución de Tipos de Archivos - Institucional'
    elif filtro == 'territorial':
        df_temp = df[df['territorial'] == True]
        titulo = 'Distribución de Tipos de Archivos - Territorial'
    else:
        df_temp = df
        titulo = 'Distribución de Tipos de Archivos - Global'
    
    # Contar extensiones
    conteo_extensiones = df_temp['extension'].value_counts().reset_index()
    conteo_extensiones.columns = ['extension', 'conteo']
    
    # Calcular porcentaje
    total = conteo_extensiones['conteo'].sum()
    conteo_extensiones['porcentaje'] = (conteo_extensiones['conteo'] / total * 100).round(1)
    
    # Clasificar como "pequeña" si es menor al threshold
    threshold = 5
    conteo_extensiones['tamaño'] = ['pequeña' if p < threshold else 'normal' for p in conteo_extensiones['porcentaje']]
    
    # Crear gráfico con la nueva paleta de colores
    fig = px.pie(
        conteo_extensiones, 
        values='conteo', 
        names='extension',
        title=titulo,
        hole=0.3,
        color_discrete_sequence=['#0A5C99', '#1E88E5', '#FEC109', '#FC9F0B']  # Nueva paleta personalizada
    )
    
    # Configurar texto
    fig.update_traces(
        textposition=["outside" if t == "pequeña" else "inside" for t in conteo_extensiones['tamaño']],
        textinfo="percent+label",
        textfont_size=12,
        pull=[0.05 if t == "pequeña" else 0 for t in conteo_extensiones['tamaño']]
    )
    
    # Diseño
    fig.update_layout(
        template='plotly_white', 
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# Función para crear gráfico de distribución por dimensiones
def crear_grafico_dimensiones(df, filtro=None):
    # Aplicar filtro si es necesario
    if filtro == 'institucional':
        df_temp = df[df['institucional'] == True]
        titulo = 'Distribución por Dimensiones - Institucional'
    elif filtro == 'territorial':
        df_temp = df[df['territorial'] == True]
        titulo = 'Distribución por Dimensiones - Territorial'
    else:
        df_temp = df
        titulo = 'Distribución por Dimensiones - Global'
    
    # Filtrar solo dimensiones clasificadas
    df_temp = df_temp[df_temp['dimensiones'] != 'Sin clasificación'].copy()
    
    # Si no hay datos, devolver mensaje de error
    if df_temp.empty:
        return None
    
    # Contar dimensiones
    conteo_dimensiones = df_temp['dimensiones'].value_counts().reset_index()
    conteo_dimensiones.columns = ['dimension', 'conteo']
    
    # Ordenar por nombre de dimensión
    conteo_dimensiones = conteo_dimensiones.sort_values('dimension')
    
    # Crear gráfico con la nueva paleta de colores
    fig = px.pie(
        conteo_dimensiones, 
        values='conteo', 
        names='dimension',
        title=titulo,
        hole=0.3,
        color_discrete_sequence=['#0A5C99', '#1E88E5', '#FEC109', '#FC9F0B']  # Nueva paleta personalizada
    )
    
    # Configurar texto
    fig.update_traces(
        textposition='auto',
        textinfo="percent+label",
        textfont_size=12
    )
    
    # Diseño
    fig.update_layout(
        template='plotly_white', 
        height=500
    )
    
    return fig

# Función para crear gráfico comparativo de extensiones por categoría
def crear_grafico_comparativo_extensiones(df):
    # Obtener top 5 extensiones
    top_ext = df['extension'].value_counts().head(5).index.tolist()
    
    # Filtrar dataframe
    df_inst = df[df['institucional'] == True]
    df_terr = df[df['territorial'] == True]
    
    # Contar extensiones por categoría
    ext_inst = df_inst[df_inst['extension'].isin(top_ext)]['extension'].value_counts()
    ext_terr = df_terr[df_terr['extension'].isin(top_ext)]['extension'].value_counts()
    
    # Completar valores faltantes con ceros
    for ext in top_ext:
        if ext not in ext_inst:
            ext_inst[ext] = 0
        if ext not in ext_terr:
            ext_terr[ext] = 0
    
    # Ordenar por el total
    total_ext = ext_inst + ext_terr
    orden = total_ext.sort_values(ascending=False).index
    
    # Crear figura con subplots
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Tipos de archivos Institucionales", "Tipos de archivos Territoriales"),
                        specs=[[{"type": "pie"}, {"type": "pie"}]])
    
    # Añadir gráficos de pastel con nuevos colores
    fig.add_trace(
        go.Pie(
            labels=orden,
            values=[ext_inst[ext] for ext in orden],
            name="Institucional",
            hole=0.4,
            marker=dict(colors=['#0A5C99', '#1E88E5', '#FEC109', '#FC9F0B'])  # Nueva paleta personalizada
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(
            labels=orden,
            values=[ext_terr[ext] for ext in orden],
            name="Territorial",
            hole=0.4,
            marker=dict(colors=['#0A5C99', '#1E88E5', '#FEC109', '#FC9F0B'])  # Nueva paleta personalizada
        ),
        row=1, col=2
    )
    
    # Actualizar diseño
    fig.update_layout(
        title_text="Comparación de Tipos de Archivos por Categoría",
        height=500,
        template="plotly_white"
    )
    
    return fig

# Función para crear heatmap de extensiones por dimensión
def crear_heatmap_extension_dimension(df):
    # Obtener top 6 extensiones
    top_ext = df['extension'].value_counts().head(6).index.tolist()
    
    # Filtrar dataframe
    df_filt = df[(df['extension'].isin(top_ext)) & (df['dimensiones'] != 'Sin clasificación')]
    
    if df_filt.empty:
        return None
    
    # Crear tabla pivote
    pivot = pd.pivot_table(
        df_filt,
        values='nombre',
        index='extension',
        columns='dimensiones',
        aggfunc='count',
        fill_value=0
    )
    
    # Crear heatmap con la paleta personalizada
    # Para heatmaps es mejor usar una escala de un solo color, así que usamos azules
    fig = px.imshow(
        pivot,
        labels=dict(x="Dimensión", y="Extensión", color="Cantidad"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale=[[0, '#E3F2FD'], [0.5, '#1E88E5'], [1, '#0A5C99']],  # Escala de azules de la paleta
        title='Distribución de Tipos de Archivos por Dimensión'
    )
    
    # Añadir valores en las celdas
    annotations = []
    for i, ext in enumerate(pivot.index):
        for j, dim in enumerate(pivot.columns):
            annotations.append(dict(
                x=dim, y=ext,
                text=str(pivot.loc[ext, dim]),
                showarrow=False,
                font=dict(color='white' if pivot.loc[ext, dim] > pivot.values.max()/2 else 'black')
            ))
    
    fig.update_layout(annotations=annotations, height=450)
    
    return fig

# Función para crear gráfico de métodos de obtención
def crear_grafico_metodos_obtencion():
    
    dfh=pd.read_excel('DataLake_registro_FIUT_UTEM.xlsx')
    dfh['METODO'].value_counts()

    dfhh={
        'nombres':[], 
        'conteo':[]
    }
    for i,j in dfh['METODO'].value_counts().items():
        dfhh['nombres'].append(i)
        dfhh['conteo'].append(j)
    dfhh=pd.DataFrame(dfhh)

    dfhh['nombres'][0]= 'Web Scrapping'
    dfhh['nombres'][1]= 'Universidad'
    dfhh['nombres'][2]= 'Descargados'

    # Datos de ejemplo (si tienes los reales, deberías usarlos)
    metodos = {
        'Web Scraping': 45,
        'Universidad': 32,
        'Descargados': 23
    }
    
    fig=px.pie(
    dfhh, 
    values='conteo', 
    names='nombres', 
    title='Distrubución métodos de obtención de los archivos',
    color_discrete_sequence=['#0A5C99', '#1E88E5', '#FEC109'],  # Nueva paleta personalizada
    hole=0.3,  # Para hacer un gráfico de dona
    )

    # Configurar texto con posiciones adaptativas
    fig.update_traces(
        textposition='auto',  # 'auto' ajusta la posición automáticamente
        textinfo='percent+label',  # Muestra porcentaje y etiqueta
        textfont_size=12,  # Tamaño de texto más grande
        rotation=270
    )

    # Mejorar el diseño
    fig.update_layout(
        template='presentation', 
        height=400,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.8,  # Posición de la leyenda
            xanchor="center",
            font=dict(size=12)
        ),
        margin=dict(l=20, r=20, t=60, b=20),  # Márgenes reducidos
        uniformtext_minsize=10,  # Tamaño mínimo de texto
        uniformtext_mode='hide'  # Ocultar texto si no hay espacio
    )
    
    return fig

def main():
    # Aplicar estilo CSS personalizado para centrar imágenes en columnas
    st.markdown("""
    <style>
        /* Centrar contenido en las columnas */
        div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Columna izquierda para una imagen (con ruta corregida)
    with col1:
        st.image("imagenes/Ministerio de Ciencias color.png", width=150)

    # Columna derecha para otra imagen (con ruta corregida)
    with col2:
        st.image("imagenes/Isologo FIU UTEM color.png", width=400)

    st.title("Proyecto FIUT 2024 UTEM")
    
    # Cargar y procesar datos
    df = cargar_datos()
    df = procesar_datos(df)
    
    if df.empty:
        st.warning("No hay datos disponibles para analizar.")
        return
    
    # Contador total de archivos
    total_archivos = len(df)
    st.markdown(f"### Levantamiento de un diagnóstico integral del territorio local y de las capacidades institucionales UTEM para la creación de un Centro Interdisciplinario en nuevas economías y tecnologías, orientado al desarrollo de localidades prioritarias de la Región Metropolitana. (CINET)")
    
    # Métricas principales
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     inst_count = df['institucional'].sum()
    #     st.metric("Archivos Institucionales", inst_count, f"{inst_count/total_archivos:.1%}")
    
    # with col2:
    #     terr_count = df['territorial'].sum()
    #     st.metric("Archivos Territoriales", terr_count, f"{terr_count/total_archivos:.1%}")
    
    # with col3:
    #     ext_count = df['extension'].nunique()
    #     st.metric("Tipos de Archivos", ext_count)

        # Métricas principales
    col1, col2= st.columns(2)
    with col1:
        inst_count = df['institucional'].sum()
        st.metric("Archivos Institucionales", inst_count, f"{inst_count/total_archivos:.1%}")
    
    with col2:
        terr_count = df['territorial'].sum()
        st.metric("Archivos Territoriales", terr_count, f"{terr_count/total_archivos:.1%}")
    
    # Pestañas para diferentes análisis
    tab1, tab2, tab3, tab4, tab5, tab6= st.tabs([
        "Vista General", 
        "Análisis por Dimensiones",
        "Insights Adicionales",
        "Mapa Geográfico",
        "Mapa Sedes",
        "Análisis de Archivos"
    ])


    # NOTA: Añadir treemap a la vista general 
    # NOTA: Los graficos de vista general pasan a análisis por Tipo
    # TAB 1: Vista General
    with tab1:
        # Puedes ajustar el tamaño del mapa según necesites
        treemap_height = 800
        
        # Función para leer el archivo HTML
        def cargar_html_treemap(ruta_html):
            try:
                with open(ruta_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return html_content
            except FileNotFoundError:
                st.error(f"No se encontró el archivo HTML del mapa en: {ruta_html}")
                return None
        
        # Ruta a tu archivo HTML (ajusta según donde esté guardado)
        ruta_mapa = "treemap_simple.html"
        
        # Cargar y mostrar el mapa
        html_treemap = cargar_html_treemap(ruta_mapa)
        if html_treemap:
            st.markdown("### Treemap de dimensiones e indicadores.")
            components.html(html_treemap, height=treemap_height)
        else:
            st.warning("No se pudo cargar el mapa. Verifica la ruta del archivo HTML.")
        

    with tab2:
        st.header("Análisis por Dimensiones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Selector para filtrar dimensiones
            filtro_dim = st.radio(
                "Seleccionar categoría para dimensiones:",
                ["Global", "Institucional", "Territorial"],
                horizontal=True
            )
            filter_dim = None if filtro_dim == "Global" else filtro_dim.lower()
            
            # Gráfico de dimensiones
            grafico_dim = crear_grafico_dimensiones(df, filter_dim)
            if grafico_dim:
                st.plotly_chart(grafico_dim, use_container_width=True, key=f"dim_{filtro_dim}_chart")       
            else:
                st.warning(f"No hay datos suficientes para mostrar dimensiones en la categoría {filtro_dim}")
        
        with col2:
            st.markdown("""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-top:35px;">
            <h4>¿Qué son las dimensiones?</h4>
            <p>Las dimensiones representan áreas funcionales o temáticas dentro de las categorías principales.
            Cada dimensión agrupa información relacionada con un aspecto específico de la gestión institucional
            o territorial, facilitando la organización y recuperación de la información.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cargar el CSV de nombres de dimensiones
            nombres_dimensiones = pd.read_csv("nombres_dimensiones.csv")
            # Crear un diccionario para mapear id a nombre
            dict_dimensiones = dict(zip(nombres_dimensiones['id_dim'], nombres_dimensiones['nombre_dim']))

            # Mostrar estadísticas por dimensión
            st.subheader("Estadísticas por Dimensión")

            # Filtrar según selección
            if filter_dim == 'institucional':
                df_stat = df[df['institucional'] == True]
            elif filter_dim == 'territorial':
                df_stat = df[df['territorial'] == True]
            else:
                df_stat = df
                
            # Calcular estadísticas de dimensiones sin "Sin clasificación"
            df_dims = df_stat[df_stat['dimensiones'] != 'Sin clasificación']

            if not df_dims.empty:
                dim_stats = df_dims['dimensiones'].value_counts()
                
                # Crear DataFrame para las estadísticas
                data = []
                for dim in dim_stats.index:
                    # Extraer el número de dimensión
                    if isinstance(dim, str) and dim.startswith('Dimensión '):
                        dim_num = int(dim.replace('Dimensión ', ''))
                    else:
                        dim_num = int(dim) if str(dim).isdigit() else 0
                    
                    # Obtener el nombre completo
                    nombre_completo = dict_dimensiones.get(dim_num, "Sin nombre")
                    
                    data.append({
                        'Número': dim_num,
                        'Dimensión': dim, 
                        'Nombre Dimensión': nombre_completo,
                        'Total Archivos': dim_stats[dim],
                        'Porcentaje': round(dim_stats[dim] / dim_stats.sum() * 100, 1)
                    })
                
                # Crear DataFrame y ordenar por número de dimensión
                dim_df = pd.DataFrame(data)
                dim_df = dim_df.sort_values('Número')
                
                # Mostrar el DataFrame sin el índice y sin la columna de número
                st.dataframe(
                    dim_df[['Dimensión', 'Nombre Dimensión', 'Total Archivos', 'Porcentaje']], 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No hay datos de dimensiones disponibles para esta selección.")
        
        # Heatmap de extensiones por dimensión
        st.subheader("Relación entre Tipos de Archivos y Dimensiones")
        
        heatmap = crear_heatmap_extension_dimension(df)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True, key="heatmap_chart")
        else:
            st.warning("No hay suficientes datos para crear el mapa de calor.")
        
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <h4>¿Qué nos muestra este mapa de calor?</h4>
        <p>Este mapa de calor muestra la concentración de diferentes tipos de archivos en cada dimensión, 
        permitiendo identificar:</p>
        <ul>
            <li>Qué formatos son más utilizados en cada dimensión</li>
            <li>Posibles patrones de uso específicos por área temática</li>
            <li>Dimensiones con mayor diversidad o especialización en formatos</li>
        </ul>
        <p>Esta información puede ser útil para entender mejor los flujos de trabajo y necesidades de 
        información en diferentes áreas de la organización.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # NOTA: Añadir metodología de trabajo
    # TAB 4: Insights Adicionales
    with tab3:
        st.header("Insights Adicionales")
        
        # Método de obtención (ejemplo)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Métodos de Obtención de Archivos")
            st.plotly_chart(crear_grafico_metodos_obtencion(), use_container_width=True, key="metodos_obtencion_chart")
        
        with col2:
            st.markdown("""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-top:35px;">
            <h4>Fuentes de información</h4>
            <p>Los archivos del Data Lake provienen de diferentes fuentes, lo que influye en su formato, 
            estructura y calidad. Las principales fuentes son:</p>
            <ul>
                <li><strong>Web Scraping:</strong> Datos extraídos automáticamente de sitios web</li>
                <li><strong>Universidad:</strong> Documentos generados internamente por la institución</li>
                <li><strong>Descargados:</strong> Archivos obtenidos de fuentes externas como portales oficiales</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Análisis de tamaño de archivos
        st.subheader("Tamaño de Archivos por Extensión")
        
        # Función para convertir tamaño a KB
        def extraer_tamano_kb(tam_str):
            try:
                if isinstance(tam_str, str):
                    partes = tam_str.split()
                    valor = float(partes[0])
                    unidad = partes[1]
                    
                    if unidad == 'B':
                        return valor / 1024
                    elif unidad == 'KB':
                        return valor
                    elif unidad == 'MB':
                        return valor * 1024
                    elif unidad == 'GB':
                        return valor * 1024 * 1024
                    else:
                        return 0
                else:
                    return 0
            except:
                return 0
        
        # Calcular tamaño en KB
        df['tamano_kb'] = df['tamano'].apply(extraer_tamano_kb)
        
        # Agrupar por extensión
        tamano_por_ext = df.groupby('extension')['tamano_kb'].agg(['mean', 'sum', 'count']).reset_index()
        tamano_por_ext.columns = ['Extensión', 'Tamaño Promedio (KB)', 'Tamaño Total (KB)', 'Cantidad']
        tamano_por_ext = tamano_por_ext.sort_values('Tamaño Total (KB)', ascending=False).head(10)
        
        # Redondear valores
        tamano_por_ext['Tamaño Promedio (KB)'] = tamano_por_ext['Tamaño Promedio (KB)'].round(2)
        tamano_por_ext['Tamaño Total (KB)'] = tamano_por_ext['Tamaño Total (KB)'].round(2)
        
        # Mostrar tabla
        st.dataframe(tamano_por_ext, use_container_width=True)
        
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <h4>Conclusiones generales</h4>
        <p>El análisis del Data Lake revela patrones importantes sobre cómo se almacena y organiza la 
        información en la organización:</p>
        <ul>
            <li>La mayor parte de los archivos son de tipo <strong>hoja de cálculo</strong>, indicando un 
            enfoque en análisis de datos cuantitativos</li>
            <li>Existe una diferencia notable entre la cantidad de archivos <strong>institucionales</strong> 
            versus <strong>territoriales</strong></li>
            <li>Cada dimensión muestra preferencias específicas por ciertos formatos, reflejando sus 
            necesidades particulares</li>
        </ul>
        <p>Esta información puede utilizarse para optimizar la gestión documental, mejorar los procesos 
        de captura de datos y facilitar el acceso a la información relevante.</p>
        </div>
        """, unsafe_allow_html=True)

    # NOTA: hablar del territorio 
    # TAB 5: Mapa Geográfico
    with tab4:
        st.header("Mapa de la Región Metropolitana")
        
        # Puedes ajustar el tamaño del mapa según necesites
        mapa_height = 600
        
        # Función para leer el archivo HTML
        def cargar_html_mapa(ruta_html):
            try:
                with open(ruta_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return html_content
            except FileNotFoundError:
                st.error(f"No se encontró el archivo HTML del mapa en: {ruta_html}")
                return None
        
        # Ruta a tu archivo HTML (ajusta según donde esté guardado)
        ruta_mapa = "mapa_rm_final.html"
        
        # Cargar y mostrar el mapa
        html_mapa = cargar_html_mapa(ruta_mapa)
        if html_mapa:
            st.markdown("Este mapa muestra las diferentes provincias y comunas de la Región Metropolitana.")
            components.html(html_mapa, height=mapa_height)
        else:
            st.warning("No se pudo cargar el mapa. Verifica la ruta del archivo HTML.")
            
        # Agregar contexto sobre el mapa
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-top:20px;">
        <h4>Acerca del mapa</h4>
        <p>Este mapa interactivo muestra la distribución territorial de la Región Metropolitana de Santiago, 
        con sus diferentes provincias identificadas por colores:</p>
        <ul>
            <li><strong>Santiago:</strong> Zona central y de mayor densidad de población</li>
            <li><strong>Cordillera:</strong> Zona este, limítrofe con la cordillera de los Andes</li>
            <li><strong>Chacabuco:</strong> Zona norte de la región</li>
            <li><strong>Maipo:</strong> Zona sur</li>
            <li><strong>Melipilla:</strong> Zona suroeste</li>
            <li><strong>Talagante:</strong> Zona oeste</li>
        </ul>
        <p>Puedes interactuar con el mapa para ver información detallada de cada comuna.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.header("Mapa de la Región Metropolitana")
        
        # Puedes ajustar el tamaño del mapa según necesites
        mapa_height = 600
        
        # Función para leer el archivo HTML
        def cargar_html_mapa(ruta_html):
            try:
                with open(ruta_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return html_content
            except FileNotFoundError:
                st.error(f"No se encontró el archivo HTML del mapa en: {ruta_html}")
                return None
        
        # Ruta a tu archivo HTML (ajusta según donde esté guardado)
        ruta_mapa = "mapa_sedes_utem.html"
        
        # Cargar y mostrar el mapa
        html_mapa = cargar_html_mapa(ruta_mapa)
        if html_mapa:
            st.markdown("Este mapa muestra las diferentes provincias y comunas de la Región Metropolitana.")
            components.html(html_mapa, height=mapa_height)
        else:
            st.warning("No se pudo cargar el mapa. Verifica la ruta del archivo HTML.")

    

    with tab6:
        
        st.header("Análisis archivos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(crear_grafico_institucional_territorial(df), use_container_width=True, key="inst_terr_chart")
            
            st.markdown("""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
            <h4>¿Qué nos muestra este gráfico?</h4>
            <p>Este gráfico muestra la distribución de archivos entre las categorías <strong>Institucional</strong> 
            y <strong>Territorial</strong>, permitiendo identificar rápidamente el balance entre estos dos tipos 
            de información en el Data Lake.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.plotly_chart(crear_grafico_extensiones(df), use_container_width=True, key="ext_general_chart")
            
            st.markdown("""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
            <h4>Tipos de archivos en el Data Lake</h4>
            <p>La distribución de tipos de archivos nos permite entender qué formatos predominan en el repositorio,
            lo que refleja los tipos de datos y documentos más utilizados en la organización.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico comparativo de extensiones por categoría
        st.header("Comparación de Tipos de Archivos por Categoría")
        st.plotly_chart(crear_grafico_comparativo_extensiones(df), use_container_width=True, key="ext_comp_chart")
        
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <h4>Diferencias entre categorías</h4>
        <p>Esta comparación permite identificar si existen patrones o preferencias diferentes en el uso de formatos 
        de archivos entre las áreas institucionales y territoriales. Esto puede reflejar diferentes necesidades
        o flujos de trabajo específicos para cada categoría.</p>
        </div>
        """, unsafe_allow_html=True)


        st.header("Análisis Detallado por Tipo de Archivo")
        
        # Selector para filtrar por categoría
        filtro_cat = st.radio(
            "Seleccionar categoría:",
            ["Global", "Institucional", "Territorial"],
            horizontal=True
        )
        filtro = None if filtro_cat == "Global" else filtro_cat.lower()
        
        # Gráfico de extensiones filtrado
        st.plotly_chart(crear_grafico_extensiones(df, filtro), use_container_width=True, key=f"ext_{filtro_cat}_chart")
        
        # Mostrar top extensiones con estadísticas
        st.subheader(f"Top 5 Extensiones - {filtro_cat}")
        
        # Filtrar según selección
        if filtro == 'institucional':
            df_temp = df[df['institucional'] == True]
        elif filtro == 'territorial':
            df_temp = df[df['territorial'] == True]
        else:
            df_temp = df
            
        # Calcular estadísticas
        top_ext = df_temp['extension'].value_counts().head(5)
        top_ext_df = pd.DataFrame({
            'Extensión': top_ext.index,
            'Cantidad': top_ext.values,
            'Porcentaje': (top_ext.values / len(df_temp) * 100).round(1)
        })
        
        # Mostrar tabla
        st.dataframe(top_ext_df, use_container_width=True)
        
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <h4>Interpretación de los tipos de archivos</h4>
        <p>Los diferentes tipos de archivos tienen propósitos específicos:</p>
        <ul>
            <li><strong>.xlsx/.xls:</strong> Hojas de cálculo para análisis de datos, registros y reportes cuantitativos</li>
            <li><strong>.pdf:</strong> Documentos formales, informes finales, documentación oficial</li>
            <li><strong>.docx/.doc:</strong> Documentos de texto, informes en proceso, documentación detallada</li>
            <li><strong>.pptx/.ppt:</strong> Presentaciones para reuniones y exposiciones</li>
            <li><strong>.csv:</strong> Datos estructurados para análisis y procesamiento</li>
        </ul>
        <p>La predominancia de ciertos formatos puede indicar el enfoque principal del trabajo en cada área.</p>
        </div>
        """, unsafe_allow_html=True)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
