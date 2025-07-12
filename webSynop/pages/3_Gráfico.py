import streamlit as st
import pandas as pd
from datetime import datetime, time
import sys
import ast
import plotly.express as px
from windrose import WindroseAxes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import stats
from utils.shared_functions import degrees_to_direction
from utils.shared_functions import get_max_observation_time
from utils.database import get_connection, run_query
from utils.shared_functions import statistical_report

st.markdown(
    """
    <h1 style='text-align: center; font-size: 24px; color: blue;'>
        Observaciones de Superficie
    </h1>
    """,
    unsafe_allow_html=True
)

graphDic = {'Barras' : ['1 hora', '08:00-19:00', '20:00-07:00', '24 horas'], 'Dispersión' : ['24 horas', '08:00-19:00', '20:00-07:00'], 'Línea (1 variable)' : ['24 horas'], 'Línea (2 variables)' : ['24 horas'], 'Histograma' : ['24 horas', '08:00-19:00', '20:00-07:00', '1 hora'], 'Caja' : ['24 horas', '08:00-19:00', '20:00-07:00'], 'Rosa de los Vientos' : ['24 horas', '08:00-19:00', '20:00-07:00']}

variableDic = {'Temperatura del aire' : ['T', 'Tq', '°C'], 'Temperatura mínima del aire' : ['Tn', 'Tnq', '°C'], 'Temperatura máxima del aire' : ['Tx', 'Txq', '°C'],
               'Temperatura del punto de rocío' : ['Td', 'Tdq', '°C'], 'Humedad Relativa' : ['HR', 'HRq', '%'], 'Índice de calor NOAA' : ['IC', 'ICq', '°C'],
               'Presión al nivel medio del mar' : ['Pnmm', 'Pnmmq', 'hPa'], 'Cambio de presión en 3h' : ['dP3', 'dP3q', 'hPa'], 'Cambio de presión en 24h' : ['dP24', 'dP24q', 'hPa'],
               'Lluvia en 1(3) hora(s)' : ['R3', 'R3q', 'mm'], 'Lluvia en 6 horas' : ['R6', 'R6q', 'mm'], 'Lluvia en 24 horas' : ['R24', 'R24q', 'mm'],
               'Velocidad del viento' : ['ff', 'ffq', 'km/h'], 'Dirección del viento' : ['dd', 'ddq', 'rumbos'], 'Racha máxima' : ['fx', 'fxq', 'km/h'], 'Nubosidad' : ['N', 'Nq', 'octas']
            }

variableDicInv = {'T': 'Temperatura del aire', 'Tn': 'Temperatura mínima del aire', 'Tx': 'Temperatura máxima del aire',
                   'Td': 'Temperatura del punto de rocío', 'HR' : 'Humedad Relativa', 'IC': 'Índice de calor',
                   'Pnmm' : 'Presión al nivel medio del mar', 'dP3' : 'Cambio de presión en 3h', 'dP24' : 'Cambio de presión en 24h',
                   'R3' : 'Lluvia en 1(3) hora(s)', 'R6' : 'Lluvia en 6 horas', 'R24' : 'Lluvia en 24 horas',
                   'ff' : 'Velocidad del viento', 'dd' : 'Dirección del viento', 'fx' : 'Racha máxima', 'N' : 'Nubosidad'
                 }

selected_hours = [1, 4, 7, 10, 13, 16, 19, 22]

stationDic_1 = {}

try:
    with open('stationDic_1.txt') as f:
        data = f.read()
        stationDic_1 = ast.literal_eval(data)
except IOError:
    sys.exit()

max_datetime = get_max_observation_time()

if max_datetime is None:
    st.error("No se pudo obtener la fecha máxima de observación")
    st.stop()  # Stop execution if no date is available

st.session_state['last_observation'] = max_datetime
try:
    max_data_year = max_datetime.year
    max_data_month = max_datetime.month
    max_data_day = max_datetime.day
    max_data_hour = max_datetime.hour
except AttributeError as e:
    st.error(f"Error al procesar la fecha: {str(e)}")
    st.stop()

st.session_state['selected_hour'] = max_data_hour

# Start and End Date initialization
start_date = datetime(max_data_year, max_data_month, max_data_day, 0, 0)
end_date = datetime(max_data_year, max_data_month, max_data_day, max_data_hour, 0)

if 'data_var_2' not in st.session_state:
    st.session_state['data_var_2'] = None
if 'data_flag_2' not in st.session_state:
    st.session_state['data_flag_2'] = None
if 'selected_schedule' not in st.session_state:
    st.session_state['selected_schedule'] = None

# Graph selection
st.session_state['selected_graph'] = st.sidebar.selectbox('Seleccione el Tipo de Gráfico', list(graphDic.keys()))

# Variable selection
if st.session_state['selected_graph'] != 'Rosa de los Vientos':
    selected_var = st.sidebar.selectbox('Seleccione la Variable 1', list(variableDic.keys()), key='var1')
    st.session_state['data_var'] = variableDic[selected_var][0]
    st.session_state['data_flag'] = variableDic[selected_var][1]
    st.session_state['data_unit'] = variableDic[selected_var][2]
    st.session_state['data_name'] = variableDicInv[st.session_state['data_var']]
    if st.session_state['selected_graph'] in ['Dispersión', 'Línea (2 variables)']:
        remaining_options = [opt for opt in list(variableDic.keys()) if opt != selected_var]
        selected_var_2 = st.sidebar.selectbox('Seleccione la Variable 2', remaining_options, key='var2', index=3) 
        st.session_state['data_var_2'] = variableDic[selected_var_2][0]
        st.session_state['data_flag_2'] = variableDic[selected_var_2][1]
        st.session_state['data_unit_2'] = variableDic[selected_var_2][2]
        st.session_state['data_name_2'] = variableDicInv[st.session_state['data_var_2']]
else:
    st.session_state['data_name'] = 'Velocidad y Dirección del viento'

# Region or province selection
st.session_state['selected_region'] = st.sidebar.selectbox('Seleccione la Región/Provincia', list(stationDic_1.keys()), key=None)

# Station selection
st.session_state['selected_station'] = st.sidebar.selectbox('Seleccione la Estación', stationDic_1[st.session_state['selected_region']])

# Start and end date selection
selected_date = st.sidebar.date_input('Seleccione la Fecha Inicial y Final', value=(start_date, end_date))

# Set end_date if omitted selection
st.session_state['start_date'] = selected_date[0]
if len(selected_date) == 1:
    st.session_state['end_date'] = st.session_state['start_date'] 
else:
    st.session_state['end_date'] = datetime(selected_date[1].year, selected_date[1].month, selected_date[1].day, 23, 0)

# Hour period selection
st.session_state['selected_period'] = st.sidebar.selectbox('Seleccione el Período', graphDic[st.session_state['selected_graph']])

# Hour selection
if st.session_state['selected_period'] == '1 hora':
    selected_time = st.sidebar.time_input('Seleccione la Hora', time(max_data_hour, 0), step=3600)
    st.session_state['selected_hour'] = selected_time.hour
else:    
    st.session_state['selected_schedule'] = st.sidebar.selectbox('Seleccione las Observaciones', ['Horarias y Trihorarias', 'Trihorarias'])

# Define the query
query = f"""
SELECT Observations.station_id,
       Stations.name,
       Stations.province,
       Stations.region,
       Observations.obs_time,
       Observations.air_temperature,
       Observations.air_temperature_flag,
       Observations.minimum_temperature,
       Observations.minimum_temperature_flag,
       Observations.maximum_temperature,
       Observations.maximum_temperature_flag,
       Observations.sea_level_pressure,
       Observations.pressure_tendency,
       Observations.pressure_change_3h,
       Observations.pressure_change_24h,
       Observations.dewpoint_temperature,
       Observations.dewpoint_temperature_flag,
       Observations.relative_humidity,
       Observations.relative_humidity_flag,
       Observations.heat_index,
       Observations.precipitation_s1,
       Observations.precipitation_s1_flag,       
       Observations.precipitation_s3,
       Observations.precipitation_s3_flag,
       Observations.precipitation_24h,
       Observations.precipitation_24h_flag,
       Observations.surface_wind_speed,
       Observations.surface_wind_speed_flag,
       Observations.surface_wind_direction_calm,
       Observations.surface_wind_direction,
       Observations.highest_gust_speed,
       Observations.highest_gust_speed_flag,
       Observations.highest_gust_direction,       
       Observations.cloud_cover
FROM Observations
INNER JOIN Stations ON Observations.station_id = Stations.station_id
WHERE Observations.obs_time >= ? AND Observations.obs_time <= ?
ORDER BY Observations.station_id, Observations.obs_time DESC
"""

if st.button('Cargar los Datos'):
    try:
        with get_connection() as conn:
            st.session_state['df'] = pd.read_sql(
                query, 
                conn, 
                params=[st.session_state['start_date'], st.session_state['end_date']]
            )

        if 'df' in st.session_state:
            # Express wind speed in km/h
            if 'surface_wind_speed' in st.session_state['df']:
                if pd.api.types.is_numeric_dtype(st.session_state['df']['surface_wind_speed']):
                    st.session_state['df'].loc[:, 'surface_wind_speed'] = (
                        st.session_state['df'].loc[:, 'surface_wind_speed'] * 3.6
                    ).round(1)
            if 'highest_gust_speed' in st.session_state['df']:
                if pd.api.types.is_numeric_dtype(st.session_state['df']['highest_gust_speed']):
                    st.session_state['df'].loc[:, 'highest_gust_speed'] = (
                        st.session_state['df'].loc[:, 'highest_gust_speed'] * 3.6
                    ).round(1)
            # Conditional sign flip (only if all required columns exist and are numeric and pressure_tendency > 5)
            df = st.session_state['df']
            cols_required = ['pressure_tendency', 'pressure_change_3h']
            # Check if all required columns exist and are numeric
            if all(col in df.columns for col in cols_required):
                if all(pd.api.types.is_numeric_dtype(df[col]) for col in cols_required):
                    # Apply the operation safely
                    mask = df['pressure_tendency'] > 5
                    df.loc[mask, 'pressure_change_3h'] = df.loc[mask, 'pressure_change_3h'] * -1
                    st.session_state['df'] = df  # Update session state

        # Rename the DataFrame columns
        st.session_state['df'].columns = [
                            'OMM',
                            'Nombre',
                            'Prov',
                            'Reg',
                            'Fecha',
                            'T',
                            'Tq',
                            'Tn',
                            'Tnq',
                            'Tx',
                            'Txq',
                            'Pnmm',
                            'a',
                            'dP3',
                            'dP24',
                            'Td',
                            'Tdq',
                            'HR',
                            'HRq',
                            'IC',
                            'R6',
                            'R6q',                            
                            'R3',
                            'R3q',
                            'R24',
                            'R24q',
                            'ff',
                            'ffq',
                            'calma',
                            'dd',
                            'fx',
                            'fxq',
                            'fxdd',                            
                            'N'
                            ]

        # insert flag columns and make them zero to make easier further processing
        st.session_state['df']['Pnmmq'] = 0
        st.session_state['df']['dP3q'] = 0
        st.session_state['df']['dP24q'] = 0
        st.session_state['df']['ICq'] = 0
        st.session_state['df']['ddq'] = 0
        st.session_state['df']['Nq'] = 0

        # Filter the DataFrame by selected_var
        if st.session_state['selected_graph'] in ['Dispersión', 'Línea (2 variables)']:
            st.session_state['filtered_df'] = st.session_state['df'][['OMM', 'Nombre', 'Reg', 'Prov', 'Fecha', st.session_state['data_var'], st.session_state['data_flag'], st.session_state['data_var_2'], st.session_state['data_flag_2']]]
        elif st.session_state['selected_graph'] == 'Rosa de los Vientos':
            st.session_state['filtered_df'] = st.session_state['df'][['OMM', 'Nombre', 'Reg', 'Prov', 'Fecha', 'ff', 'ffq', 'calma', 'dd', 'ddq']]
        else:
            st.session_state['filtered_df'] = st.session_state['df'][['OMM', 'Nombre', 'Reg', 'Prov', 'Fecha', st.session_state['data_var'], st.session_state['data_flag']]]    

        # Filter the DataFrame by selected_region
        if st.session_state['selected_region'] == 'Cuba':
            st.session_state['filtered_df'] = st.session_state['filtered_df'][((st.session_state['filtered_df']['Reg'] == 'OCC') | (st.session_state['filtered_df']['Reg'] == 'CEN') | (st.session_state['filtered_df']['Reg'] == 'OTE')) ]
        elif st.session_state['selected_region'] == 'OCC':
            st.session_state['filtered_df'] = st.session_state['filtered_df'][(st.session_state['filtered_df']['Reg'] == 'OCC')]
        elif st.session_state['selected_region'] == 'CEN':
            st.session_state['filtered_df'] = st.session_state['filtered_df'][(st.session_state['filtered_df']['Reg'] == 'CEN')]
        elif st.session_state['selected_region'] == 'OTE':
            st.session_state['filtered_df'] = st.session_state['filtered_df'][(st.session_state['filtered_df']['Reg'] == 'OTE')]
        else:
            st.session_state['filtered_df'] = st.session_state['filtered_df'][(st.session_state['filtered_df']['Prov'] == st.session_state['selected_region'])]    

        # Filter the DataFrame by selected_station
        if st.session_state['selected_station'] != 'Todas':
            st.session_state['filtered_df'] = st.session_state['filtered_df'][(st.session_state['filtered_df']['Nombre'] == st.session_state['selected_station'])]

        # Filter the DataFrame by selected_period
        if st.session_state['selected_period'] == '1 hora':
            st.session_state['filtered_df'] = st.session_state['filtered_df'][st.session_state['filtered_df']['Fecha'].dt.hour == st.session_state['selected_hour']]
        elif st.session_state['selected_period'] == '08:00-19:00':
            st.session_state['filtered_df'] = st.session_state['filtered_df'][(st.session_state['filtered_df']['Fecha'].dt.hour >= 8) & (st.session_state['filtered_df']['Fecha'].dt.hour <= 19)]
        elif st.session_state['selected_period'] == '20:00-07:00':
            st.session_state['filtered_df'] = st.session_state['filtered_df'][(st.session_state['filtered_df']['Fecha'].dt.hour <= 7) | (st.session_state['filtered_df']['Fecha'].dt.hour >= 20)]

        # Filter the DataFrame by selected_schedule
        if st.session_state['selected_schedule'] == 'Trihorarias':
            st.session_state['filtered_df'] = st.session_state['filtered_df'][st.session_state['filtered_df']['Fecha'].dt.hour.isin(selected_hours)]

        # Define custom orders for each column
        province_order = ['IJU', 'PRI', 'ART', 'MAY', 'HAB', 'MTZ', 'VCL', 'CFG', 'SSP', 'CAV', 'CMG', 'LTU','HLG', 'GRA', 'SCU', 'GMO']
        region_order = ['OCC', 'CEN', 'OTE']
        st.session_state['filtered_df'].loc[:, 'Prov'] = pd.Categorical(st.session_state['filtered_df']['Prov'], categories=province_order, ordered=True)
        st.session_state['filtered_df'].loc[:, 'Reg'] = pd.Categorical(st.session_state['filtered_df']['Reg'], categories=region_order, ordered=True)

        st.session_state['filtered_df'] = (st.session_state['filtered_df'].assign(
                Prov=lambda x: pd.Categorical(x['Prov'], categories=province_order, ordered=True),
                Reg=lambda x: pd.Categorical(x['Reg'], categories=region_order, ordered=True)
            )
            .sort_values(['Reg', 'Prov'])
        )

        # Find Out min and max data date
        st.session_state['min_date'] = st.session_state['filtered_df']['Fecha'].min()
        st.session_state['max_date'] = st.session_state['filtered_df']['Fecha'].max()

        # Write date and time information of graph
        if st.session_state['selected_period'] == '1 hora':
            if {st.session_state['min_date']} == {st.session_state['max_date']}:
                st.write(f"Datos de {st.session_state['data_name']} del {st.session_state['min_date']} -- Período: {st.session_state['selected_period']}")
            else:
                st.write(f"Datos de {st.session_state['data_name']} desde {st.session_state['min_date']} hasta {st.session_state['max_date']} -- Período: {st.session_state['selected_period']} -- Hora: {st.session_state['selected_hour']}:00")
        else:
            st.write(f"Datos de {st.session_state['data_name']} desde {st.session_state['min_date']} hasta {st.session_state['max_date']} -- Período: {st.session_state['selected_period']}")

        # Write site data information of graph
        st.write(f"Región/Provincia: {st.session_state['selected_region']} -- Estación: {st.session_state['selected_station']}")

        # Create new dataframe with valid data and count total data, missing data and invalid data 
        st.session_state['all_data'] = len(st.session_state['filtered_df'])
        if st.session_state['selected_graph'] == 'Rosa de los Vientos':
            st.session_state['missing_data'] = st.session_state['filtered_df']['ff'].isna().sum()
            st.session_state['filtered_df_valid'] = st.session_state['filtered_df'][(st.session_state['filtered_df']['ffq'] <= 4)]
            st.session_state['invalid_data'] = len(st.session_state['filtered_df'][(st.session_state['filtered_df']['ffq'] > 4)])
            wind_data = len(st.session_state['filtered_df_valid'])
            st.session_state['calm'] = st.session_state['filtered_df_valid']['calma'].sum()
            st.session_state['filtered_df_valid'] = st.session_state['filtered_df_valid'].dropna()
            wind_direction = st.session_state['filtered_df_valid']['dd'].tolist()
            wind_speed = st.session_state['filtered_df_valid']['ff'].tolist()
        else:
            st.session_state['missing_data'] = st.session_state['filtered_df'][st.session_state['data_var']].isna().sum()
            st.session_state['filtered_df_valid'] = st.session_state['filtered_df'][st.session_state['filtered_df'][st.session_state['data_flag']] <= 4]
            st.session_state['invalid_data'] = len( st.session_state['filtered_df'][st.session_state['filtered_df'][st.session_state['data_flag']] > 4])
            # Find out the axis title for selected_graph
            key = next((k for k, v in variableDic.items() if v[0] == st.session_state['data_var']), None)
            if st.session_state['selected_graph'] in ['Dispersión', 'Línea (2 variables)']:
                st.session_state['filtered_df_valid'] = st.session_state['filtered_df_valid'][st.session_state['filtered_df_valid'][st.session_state['data_flag_2']] <= 4]        
                st.session_state['invalid_data'] = st.session_state['all_data'] - len(st.session_state['filtered_df_valid']) 
                key_2 = next((k for k, v in variableDic.items() if v[0] == st.session_state['data_var_2']), None)

        if st.session_state['selected_graph'] == 'Rosa de los Vientos':
            st.write(f"Total de observaciones: {st.session_state['all_data']} -- Casos de Calma: {st.session_state['calm']} = {st.session_state['calm']/wind_data:.1%} -- Sin la variable: {st.session_state['missing_data']} -- Con valores inválidos: {st.session_state['invalid_data']} = {st.session_state['invalid_data']/st.session_state['all_data']:.1%}")
        else:
            st.write(f"Total de observaciones: {st.session_state['all_data']} -- Sin la variable: {st.session_state['missing_data']} -- Con valores inválidos: {st.session_state['invalid_data']} = {st.session_state['invalid_data']/st.session_state['all_data']:.1%}")

        # Create a chart of selected_graph type with valid data
        if st.session_state['selected_graph'] == 'Barras':
            if st.session_state['selected_region'] == 'Cuba':    
                fig = px.bar(st.session_state['filtered_df_valid'], x='Nombre', y=st.session_state['data_var'], color='Reg')
            elif st.session_state['selected_region'] in ['OCC', 'CEN', 'OTE']:
                fig = px.bar(st.session_state['filtered_df_valid'], x='Nombre', y=st.session_state['data_var'], color='Prov')
            else:
                fig = px.bar(st.session_state['filtered_df_valid'], x='Nombre', y=st.session_state['data_var'], color='Nombre')
            fig.update_layout(
                title='Diagrama de Barras',
                hovermode='x unified',
                xaxis_title='Estación',
                yaxis_title= key
                )
            st.plotly_chart(fig)
        elif st.session_state['selected_graph'] == 'Dispersión':
            fig = px.scatter(
                st.session_state['filtered_df_valid'],
                x=st.session_state['data_var'],
                y=st.session_state['data_var_2'],
                color=st.session_state['data_var'],
                hover_data={'Fecha' : '|%Y-%m-%d %H:00',
                            st.session_state['data_var'] : ':.1f',
                            st.session_state['data_var_2'] : ':.1f'
                        },
                color_continuous_scale=px.colors.sequential.Bluered,
                trendline='ols',
                width=800,
                height=500
            )
            fig.update_layout(
                title='Diagrama de Dispersión',
                hovermode='x unified',
                xaxis_title=key,
                yaxis_title=key_2,
            )
            st.plotly_chart(fig)
        elif st.session_state['selected_graph'] == 'Línea (1 variable)':
            if st.session_state['selected_station'] == 'Todas':
                st.warning('Para crear el Diagrama de Líneas debe seleccionar una estación. Esperando por su elección...')
            else:
                st.session_state['filtered_df_valid'] = st.session_state['filtered_df_valid'].sort_values(by='Fecha')
                fig = px.line(st.session_state['filtered_df_valid'], x='Fecha', y=st.session_state['data_var'], markers=True)
                fig.update_xaxes(
                    tickformat='%Y-%m-%d',
                    tickangle=45
                    )
                fig.update_traces(
                    hovertemplate=f"<b>Fecha</b>: %{{x|%Y-%m-%d %H:%M}}<br><b>{st.session_state['data_var']}</b>: %{{y:.1f}}<extra></extra>"
                )
                fig.update_layout(
                    title='Diagrama de Línea (1 variable)',
                    hovermode='x unified',
                    xaxis_title='Fecha',
                    yaxis_title=key,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    height=600,
                    margin=dict(l=50, r=50, b=100, t=100)
                )
                st.plotly_chart(fig)
        elif st.session_state['selected_graph'] == 'Línea (2 variables)':
            if st.session_state['selected_station'] == 'Todas':
                st.warning('Para crear el Diagrama de Líneas debe seleccionar una estación. Esperando por su elección...')
            else:
                st.session_state['filtered_df_valid'] = st.session_state['filtered_df_valid'].sort_values(by='Fecha')
                fig = px.line(title=f'Diagrama de Línea (2 variables)')
                fig.update_xaxes(
                    tickformat='%Y-%m-%d',
                    tickangle=45
                    )
                fig.add_scatter(
                    x=st.session_state['filtered_df_valid']['Fecha'],
                    y=st.session_state['filtered_df_valid'][st.session_state['data_var']],
                    name=key,
                    mode='lines+markers',
                    marker=dict(
                        symbol='circle',
                        size=6),
                    line=dict(color='red', width=2),                            
                    hovertemplate=f"<b>Fecha</b>: %{{x|%Y-%m-%d %H:%M}}<br><b>{st.session_state['data_var']}</b>: %{{y:.1f}}<extra></extra>",
                )
                fig.add_scatter(
                    x=st.session_state['filtered_df_valid']['Fecha'],
                    y=st.session_state['filtered_df_valid'][st.session_state['data_var_2']],
                    name=key_2,
                    mode='lines+markers',
                    marker=dict(
                        symbol='diamond',
                        size=6),
                    line=dict(color='blue', width=2),
                    hovertemplate=f"<b>Fecha</b>: %{{x|%Y-%m-%d %H:%M}}<br><b>{st.session_state['data_var_2']}</b>: %{{y:.1f}}<extra></extra>",
                    yaxis='y2'
                )
                fig.update_layout(
                    hovermode='x unified',
                    xaxis_title='Fecha',
                    yaxis=dict(
                        title=key,
                        side='left',
                        range=[st.session_state['filtered_df_valid'][st.session_state['data_var']].min() * 0.95, st.session_state['filtered_df_valid'][st.session_state['data_var']].max() * 1.05]
                    ),
                    yaxis2=dict(
                        title=key_2,
                        overlaying='y',
                        side='right',
                        range=[st.session_state['filtered_df_valid'][st.session_state['data_var_2']].min() * 0.95, st.session_state['filtered_df_valid'][st.session_state['data_var_2']].max() * 1.05]
                    ),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    height=600,
                    margin=dict(l=50, r=50, b=100, t=100)
        )
                st.plotly_chart(fig)
        elif st.session_state['selected_graph'] == 'Histograma':
            data = st.session_state['filtered_df_valid'][st.session_state['data_var']]
            fig = px.histogram(x=data, nbins=20, marginal='box', title='Histograma con Diagrama de Caja')
            # Calculate percentage for hover
            total = len(data)
            fig.data[0].customdata = (fig.data[0].x / total) * 100
            fig.update_traces(
                hovertemplate='<b>%{xaxis.title.text}:</b> %{x}<br>' +
                            '<b>Conteo:</b> %{y}<br>' +
                            '<b>Porcentaje:</b> %{customdata:.1f}%<extra></extra>',
                marker=dict(line=dict(width=1, color='white'))
            )
            fig.update_traces(
                hovertemplate='<b>Boxplot</b><br>Value: %{x}<br>%{text}',
                text=['Median' if x == np.median(data) else 
                    'Q1' if x == np.percentile(data, 25) else
                    'Q3' if x == np.percentile(data, 75) else
                    'Min' if x == np.min(data) else 'Max' for x in fig.data[1].x],
                marker=dict(color='lightblue'),        
                selector=dict(type='box')
            )
            fig.update_layout(
                hovermode='x unified',
                height=600,
                showlegend=False,
                xaxis_title=key,
                yaxis_title='Frecuencia'
            )
            st.plotly_chart(fig)
        elif st.session_state['selected_graph'] == 'Caja':
            if st.session_state['selected_region'] == 'Cuba':
                fig = px.box(st.session_state['filtered_df_valid'], x='Nombre', y=st.session_state['data_var'], color='Reg')
            elif st.session_state['selected_region'] in ['Occidente', 'Centro', 'Oriente']:
                fig = px.box(st.session_state['filtered_df_valid'], x='Nombre', y=st.session_state['data_var'], color='Prov')
            else:
                fig = px.box(st.session_state['filtered_df_valid'], x='Nombre', y=st.session_state['data_var'], color='Nombre')
            fig.update_layout(
                title='Diagrama de Caja',
                xaxis_title='Estación',
                yaxis_title= key
                )
            st.plotly_chart(fig)
        elif st.session_state['selected_graph'] == 'Rosa de los Vientos':
            if len(wind_direction) != 0:
                chart = plt.figure(figsize=(10, 8), dpi=100)
                ax = WindroseAxes.from_ax(fig=chart)
                ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, bins=np.linspace(0, max(wind_speed), 7), edgecolor='black', cmap=cm.plasma)
                ax.set_legend(title='Velocidad (km/h)', bbox_to_anchor=(1.1, 1), loc='upper left')
                ax.set_theta_direction('counterclockwise')
                ax.set_theta_zero_location('E')

                ax.set_title(
                    'Rosa de los Vientos',  # Title text
                    fontdict={
                        'fontsize': 12,        # Font size
                        'fontweight': 'bold',  # Font weight (normal, bold, semibold)
                        'fontfamily': 'serif', # Font family (serif, sans-serif, monospace)
                        'color': 'black',    # Text color
                        'verticalalignment': 'baseline'  # Vertical alignment
                    },
                    pad=20,                    # Padding between title and plot
                    loc='left',                # Horizontal alignment (left, center, right)
                    y=1.02                     # Vertical position adjustment
                )

                st.pyplot(chart)

        # Data raw
        with st.expander('Datos del Gráfico'):
            st.dataframe(st.session_state['filtered_df_valid'], hide_index=True)

        # Data summary
        if st.session_state['selected_graph'] == 'Dispersión':
            with st.expander('Resumen Estadístico'):
                var1 = st.session_state['data_var']
                var2 = st.session_state['data_var_2']
                st.session_state['stats_df'] = st.session_state['filtered_df_valid'][[var1, var2]].describe().T
                st.session_state['stats_df']['skewness'] = [stats.skew(st.session_state['filtered_df_valid'][st.session_state['data_var']]), stats.skew(st.session_state['filtered_df_valid'][st.session_state['data_var_2']])]
                st.session_state['stats_df']['kurtosis'] = [stats.kurtosis(st.session_state['filtered_df_valid'][st.session_state['data_var']]), stats.kurtosis(st.session_state['filtered_df_valid'][st.session_state['data_var_2']])]
                pearson = st.session_state['filtered_df_valid'][var1].corr(st.session_state['filtered_df_valid'][var2])
                spearman = st.session_state['filtered_df_valid'][var1].corr(st.session_state['filtered_df_valid'][var2], method='spearman')
                st.write(f"Análisis de regresión -- Pearson r: {pearson:.3f} -- Spearman ρ: {spearman:.3f}")
                column_config = {
                    "count": st.column_config.NumberColumn("Count", format="%.0f"),
                    "mean": st.column_config.NumberColumn("Mean", format="%.1f"),
                    "std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                    "min": st.column_config.NumberColumn("Minimum", format="%.1f"),
                    "25%": st.column_config.NumberColumn("25th %ile", format="%.1f"),
                    "50%": st.column_config.NumberColumn("Median", format="%.1f"),
                    "75%": st.column_config.NumberColumn("75th %ile", format="%.1f"),
                    "max": st.column_config.NumberColumn("Maximum", format="%.1f"),
                    "skewness": st.column_config.NumberColumn("Skew", format="%.2f"),
                    "kurtosis": st.column_config.NumberColumn("Kurt", format="%.2f"),
                }
                st.session_state['stats_df'] = st.session_state['stats_df'].reset_index()
                st.data_editor(
                    st.session_state['stats_df'],
                    column_config=column_config,
                    hide_index=True,
                    disabled=True
                )
        elif st.session_state['selected_graph'] == 'Línea (2 variables)':
            with st.expander('Resumen Estadístico'):
                var1 = st.session_state['data_var']
                var2 = st.session_state['data_var_2']
                st.session_state['stats_df'] = st.session_state['filtered_df_valid'][[var1, var2]].describe().T
                column_config = {
                    "count": st.column_config.NumberColumn("Count", format="%.0f"),
                    "mean": st.column_config.NumberColumn("Mean", format="%.1f"),
                    "std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                    "min": st.column_config.NumberColumn("Minimum", format="%.1f"),
                    "25%": st.column_config.NumberColumn("25th %ile", format="%.1f"),
                    "50%": st.column_config.NumberColumn("Median", format="%.1f"),
                    "75%": st.column_config.NumberColumn("75th %ile", format="%.1f"),
                    "max": st.column_config.NumberColumn("Maximum", format="%.1f")
                }
                st.session_state['stats_df'] = st.session_state['stats_df'].reset_index()
                st.data_editor(
                    st.session_state['stats_df'],
                    column_config=column_config,
                    hide_index=True,
                    disabled=True
                )
        else:
            with st.expander('Resumen Estadístico'):
                if st.session_state['selected_graph'] == 'Rosa de los Vientos':
                    st.session_state['data_var'] = 'ff'
                st.session_state['stats_df'] = st.session_state['filtered_df_valid'][[st.session_state['data_var']]].describe().T
                column_config = {
                    "count": st.column_config.NumberColumn("Count", format="%.0f"),
                    "mean": st.column_config.NumberColumn("Mean", format="%.1f"),
                    "std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                    "min": st.column_config.NumberColumn("Minimum", format="%.1f"),
                    "25%": st.column_config.NumberColumn("25th %ile", format="%.1f"),
                    "50%": st.column_config.NumberColumn("Median", format="%.1f"),
                    "75%": st.column_config.NumberColumn("75th %ile", format="%.1f"),
                    "max": st.column_config.NumberColumn("Maximum", format="%.1f")
                }
                st.session_state['stats_df'] = st.session_state['stats_df'].reset_index()
                st.data_editor(
                    st.session_state['stats_df'],
                    column_config=column_config,
                    hide_index=True,
                    disabled=True
                )

        if (st.session_state['data_var'] != 'dd' and st.session_state['data_var'] != 'N'):

            with st.expander('Informe de Valores Extremos'):
                if st.session_state['selected_graph'] == 'Rosa de los Vientos':
                    st.session_state['data_var'] = 'ff'
                    st.session_state['data_flag'] = 'ffq'
                    st.session_state['data_unit'] = 'km/h'
                    st.session_state['data_name'] = 'Velocidad del viento'    

                if st.session_state['data_var'] in ['T', 'Tn', 'Tx', 'Pnmm', 'dP3', 'dP24', 'Td', 'HR', 'IC']:

                    report = statistical_report(
                        df=st.session_state['filtered_df_valid'],
                        var=st.session_state['data_var'],
                        var_str=st.session_state['data_name'],
                        unit_str=st.session_state['data_unit'],
                        nulled_count_var=st.session_state['invalid_data']
                    )

                elif st.session_state['data_var'] in ['R3', 'R6', 'R24', 'ff', 'fx']:
                    
                    report = statistical_report(
                                df=st.session_state['filtered_df_valid'],
                                var=st.session_state['data_var'],
                                var_str=st.session_state['data_name'],
                                unit_str=st.session_state['data_unit'],
                                nulled_count_var=st.session_state['invalid_data'],
                                find_min=False
                            )

                # Display nulled values info if exists
                if report['nulled_info']:
                    st.info(report['nulled_info'])

                # Display min values if calculated
                if report['min'] is not None:
                    min_value, min_stations_dates = report['min']
                    if min_value is not None and not np.isnan(min_value):
                        st.write(f"Mínimo de {report['var_str']}: {min_value} {report['unit_str']} observado en:")
                        for station, date in min_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                # Display max values if calculated
                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                if st.session_state['selected_graph'] in ['Dispersión', 'Línea (2 variables)']:

                    if st.session_state['data_var_2'] in ['T', 'Tn', 'Tx', 'Pnmm', 'dP3', 'dP24', 'Td', 'HR', 'IC']:

                        report = statistical_report(
                            df=st.session_state['filtered_df_valid'],
                            var=st.session_state['data_var_2'],
                            var_str=st.session_state['data_name_2'],
                            unit_str=st.session_state['data_unit_2'],
                            nulled_count_var=st.session_state['invalid_data']
                        )
                
                    elif st.session_state['data_var_2'] in ['R3', 'R6', 'R24', 'ff', 'fx']:
                        
                        report = statistical_report(
                                    df=st.session_state['filtered_df_valid'],
                                    var=st.session_state['data_var_2'],
                                    var_str=st.session_state['data_name_2'],
                                    unit_str=st.session_state['data_unit_2'],
                                    nulled_count_var=st.session_state['invalid_data'],
                                    find_min=False
                                )

                    # Display nulled values info if exists
                    if report['nulled_info']:
                        st.info(report['nulled_info'])

                    # Display min values if calculated
                    if report['min'] is not None:
                        min_value, min_stations_dates = report['min']
                        if min_value is not None and not np.isnan(min_value):
                            st.write(f"Mínimo de {report['var_str']}: {min_value} {report['unit_str']} observado en:")
                            for station, date in min_stations_dates:
                                st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                    # Display max values if calculated
                    if report['max'] is not None:
                        max_value, max_stations_dates = report['max']
                        if max_value is not None and not np.isnan(max_value):
                            st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                            for station, date in max_stations_dates:
                                st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")