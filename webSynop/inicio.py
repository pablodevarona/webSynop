import streamlit as st
from utils.shared_functions import degrees_to_direction

st.set_page_config(page_title='Monitor de Observaciones  de Superficie')
st.markdown(
    """
    <h1 style='text-align: center; font-size: 24px; color: blue;'>
        Monitor de Observaciones de Superficie
    </h1>
    """,
    unsafe_allow_html=True
)
st.sidebar.success('Seleccione una página')
st.markdown(
    """
El Monitor de Observaciones de Superficie le permite visualizar los datos de la manera más conveniente para su análisis\n
En la página Mapa puede analizar espacialmente los datos filtrados por variable, fecha y hora. Con la herramienta tooltip puede desplegar una lista de los valores observados por la estación sobre la que puso el mouse. 
Se incluye una tabla con los datos ploteados en el mapa y, si procede, una tabla estadística de las variables numéricas y las estaciones en las que se observaron los valores mínimo y máximo\n
En la página Tabla puede analizar los datos tabulados filtrados por región/provincia/estación, fecha y hora para las variables más importantes. 
Se incluye una tabla estadística de las variables numéricas y un reporte de valores mínimo y máximo con las estaciones y fechas en las que se observaron. 
Además, se presenta un reporte de calidad de las variables observadas\n
En la página Gráfico puede analizar los datos filtrados por variable, región/provincia/estación, fecha, hora y periodo con diferentes tipos de gráficos. 
Se incluye una tabla con los datos ploteados en el mapa y, si procede, una tabla estadística para las variables numéricas y las estaciones en las que se observaron los valores mínimo y máximo
"""
)