import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, time
import time as time_2
import sys
import ast
from utils.shared_functions import degrees_to_direction
from utils.shared_functions import get_max_observation_time
from utils.shared_functions import statistical_report
from utils.database import run_query, get_connection

st.set_page_config(layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; font-size: 24px; color: blue;'>
        Observaciones de Superficie
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
.pydeck-tooltip {
    max-width: 300px !important;
}
</style>
""", unsafe_allow_html=True
)

variableDic = {'Temperatura del aire' : ['T', 'Tq', '°C'], 'Temperatura mínima del aire' : ['Tn', 'Tnq', '°C'], 'Temperatura máxima del aire' : ['Tx', 'Txq', '°C'],
               'Temperatura del punto de rocío' : ['Td', 'Tdq', '°C'], 'Humedad Relativa' : ['HR', 'HRq', '%'], 'Índice de calor NOAA' : ['IC', 'ICq', '°C'],
               'Presión al nivel medio del mar' : ['Pnmm', 'Pnmmq', 'hPa'], 'Cambio de presión en 3h' : ['dP3', 'dP3q', 'hPa'], 'Cambio de presión en 24h' : ['dP24', 'dP24q', 'hPa'],
               'Lluvia en 1(3) hora(s)' : ['R3', 'R3q', 'mm'], 'Lluvia en 6 horas' : ['R6', 'R6q', 'mm'], 'Lluvia en 24 horas' : ['R24', 'R24q', 'mm'], 'Tiempo' : ['ww', 'wwq', 'código'],
               'Velocidad del viento' : ['ff', 'ffq', 'km/h'], 'Dirección del viento' : ['dd', 'ddq', 'rumbos'], 'Racha máxima' : ['fx', 'fxq', 'km/h'], 'Nubosidad' : ['N', 'Nq', 'octas']
            }

variableDicInv = {'T': 'Temperatura del aire', 'Tn': 'Temperatura mínima del aire', 'Tx': 'Temperatura máxima del aire',
                   'Td': 'Temperatura del punto de rocío', 'HR' : 'Humedad Relativa', 'IC': 'Índice de calor',
                   'Pnmm' : 'Presión al nivel medio del mar', 'dP3' : 'Cambio de presión en 3h', 'dP24' : 'Cambio de presión en 24h',
                   'R3' : 'Lluvia en 1(3) hora(s)', 'R6' : 'Lluvia en 6 horas', 'R24' : 'Lluvia en 24 horas', 'ww' : 'Tiempo',
                   'ff' : 'Velocidad del viento', 'dd' : 'Dirección del viento', 'fx' : 'Racha máxima', 'N' : 'Nubosidad'
                 }

stationDic_0 = {}

try:
    with open('stationDic_0.txt') as f:
        data = f.read()
        stationDic_0 = ast.literal_eval(data)
except IOError:
    sys.exit()

st.session_state['station_df'] = pd.DataFrame(stationDic_0)

max_datetime = get_max_observation_time()

if max_datetime is None:
    st.error("No se pudo obtener la fecha máxima de observación")
    st.stop()  # Stop execution if no date is available

try:
    max_data_year = max_datetime.year
    max_data_month = max_datetime.month
    max_data_day = max_datetime.day
    max_data_hour = max_datetime.hour
except AttributeError as e:
    st.error(f"Error al procesar la fecha: {str(e)}")
    st.stop()

max_date = datetime(max_data_year, max_data_month, max_data_day)

selected_date = st.sidebar.date_input('Seleccione la Fecha', max_date)
st.session_state['selected_date'] = selected_date

selected_time = st.sidebar.time_input('Seleccione la Hora', time(max_data_hour, 0), step=3600)
st.session_state['selected_hour'] = selected_time.hour

selected_datetime = datetime.combine(selected_date, selected_time)

if selected_datetime > max_datetime:
    st.session_state['obs_date'] = max_datetime
    datetime_valid = False
    # Create a placeholder for the alert message
    alert_placeholder = st.empty()
    # Display an alert message
    alert_placeholder.warning(f"La fecha-hora seleccionada no puede ser mayor que {max_datetime.strftime('%Y-%m-%d %H:%M')} Hora Local (sin Horario de Verano)")
    # Wait for 3 seconds before "deleting" the alert
    time_2.sleep(3)
    # Clear the alert message
    alert_placeholder.empty()
else:
    st.session_state['obs_date'] = selected_datetime
    datetime_valid = True

selected_var = st.sidebar.selectbox('Seleccione una variable', list(variableDic.keys()))
st.session_state['data_var'] = variableDic[selected_var][0]
st.session_state['data_flag'] = variableDic[selected_var][1]
st.session_state['data_unit'] = variableDic[selected_var][2]
st.session_state['data_name'] = variableDicInv[st.session_state['data_var']]

# Define the query
query = f"""
SELECT Observations.station_id,
        Observations.obs_time,
        Observations.aws,
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
        Observations.precipitation_s3,
        Observations.precipitation_s3_flag,
        Observations.precipitation_s1,
        Observations.precipitation_s1_flag,
        Observations.precipitation_24h,
        Observations.precipitation_24h_flag,
        Observations.surface_wind_speed,
        Observations.surface_wind_speed_flag,
        Observations.surface_wind_direction_calm,
        Observations.surface_wind_direction,
        Observations.present_weather,
        PresentWeather.present_weather AS present_weather_description,
        Observations.highest_gust_speed,
        Observations.highest_gust_speed_flag,
        Observations.highest_gust_direction,
        Observations.highest_gust_date,
        Observations.cloud_cover
FROM Observations
LEFT JOIN PresentWeather ON Observations.present_weather = PresentWeather.present_weather_id
WHERE Observations.obs_time = ?
ORDER BY Observations.station_id
"""

if (st.button('Cargar los Datos') and datetime_valid):
    try:
        with get_connection() as conn:
            st.session_state['obs_df'] = pd.read_sql(
                query, 
                conn, 
                params=[st.session_state['obs_date']]
            )
        
        if 'obs_df' in st.session_state:
            # Express wind speed in km/h
            if 'surface_wind_speed' in st.session_state['obs_df']:
                if pd.api.types.is_numeric_dtype(st.session_state['obs_df']['surface_wind_speed']):
                    st.session_state['obs_df'].loc[:, 'surface_wind_speed'] = (
                        st.session_state['obs_df'].loc[:, 'surface_wind_speed'] * 3.6
                    ).round(1)
            if 'highest_gust_speed' in st.session_state['obs_df']:
                if pd.api.types.is_numeric_dtype(st.session_state['obs_df']['highest_gust_speed']):
                    st.session_state['obs_df'].loc[:, 'highest_gust_speed'] = (
                        st.session_state['obs_df'].loc[:, 'highest_gust_speed'] * 3.6
                    ).round(1)
            # Express wind direction in course
            if 'surface_wind_direction' in st.session_state['obs_df']:
                numeric_mask = st.session_state['obs_df']['surface_wind_direction'].apply(lambda x: isinstance(x, (int, float)))
                st.session_state['obs_df'].loc[numeric_mask, 'surface_wind_direction'] = st.session_state['obs_df'].loc[numeric_mask, 'surface_wind_direction'].apply(degrees_to_direction)
            if 'highest_gust_direction' in st.session_state['obs_df']:
                numeric_mask = st.session_state['obs_df']['highest_gust_direction'].apply(lambda x: isinstance(x, (int, float)))
                st.session_state['obs_df'].loc[numeric_mask, 'highest_gust_direction'] = st.session_state['obs_df'].loc[numeric_mask, 'highest_gust_direction'].apply(degrees_to_direction)        
            # Conditional sign flip (only if all required columns exist and are numeric and pressure_tendency > 5)
            df = st.session_state['obs_df']
            cols_required = ['pressure_tendency', 'pressure_change_3h']
            # Check if all required columns exist and are numeric
            if all(col in df.columns for col in cols_required):
                if all(pd.api.types.is_numeric_dtype(df[col]) for col in cols_required):
                    # Apply the operation safely
                    mask = df['pressure_tendency'] > 5
                    df.loc[mask, 'pressure_change_3h'] = df.loc[mask, 'pressure_change_3h'] * -1
                    st.session_state['obs_df'] = df  # Update session state

        # Merge station and observation dataframes 
        st.session_state['merged_df'] = pd.merge(st.session_state['station_df'], st.session_state['obs_df'], on='station_id', how='left')

        st.session_state['merged_df'].columns = [
                            'OMM',
                            'Nombre',
                            'Prov',
                            'Reg',
                            'longitude',
                            'latitude',
                            'altura',
                            'Fecha',
                            'EMA',
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
                            'R3',
                            'R3q',
                            'R6',
                            'R6q',
                            'R24',
                            'R24q',
                            'ff',
                            'ffq',
                            'calma',
                            'dd',
                            'ww',
                            'wwd',
                            'fx',
                            'fxq',
                            'fxdd',
                            'fxf',
                            'N',
                            ]

        # insert flag columns and make them zero to make easier further processing
        st.session_state['merged_df']['ICq'] = 0
        st.session_state['merged_df']['Pnmmq'] = 0
        st.session_state['merged_df']['dP3q'] = 0
        st.session_state['merged_df']['dP24q'] = 0
        st.session_state['merged_df']['ddq'] = 0
        st.session_state['merged_df']['wwq'] = 0
        st.session_state['merged_df']['Nq'] = 0

        # remove invalid data values
        condition = (
                    st.session_state['merged_df'][st.session_state['data_flag']].isnull() | 
                    (st.session_state['merged_df'][st.session_state['data_flag']] <= 4)
                    )
        # Apply the filter
        st.session_state['merged_df'] = st.session_state['merged_df'][condition]

        st.write(f"Variable {selected_var}, en {st.session_state['data_unit']} -- Fecha {st.session_state['obs_date'].strftime('%Y-%m-%d %H:%M')} Hora Local (sin Horario de Verano)")

        required_columns = ['OMM', 'Nombre', 'latitude', 'longitude', 'altura', 
                        'T', 'Tn', 'Tx', 'Td', 'HR', 'IC', 'Pnmm', 'ff', 'dd',
                        'R3', 'R6', 'R24', 'N', 'fx', 'fxdd', 'wwd']

        assert all(col in st.session_state['merged_df'].columns for col in required_columns)

        # Convert null values to JavaScript null
        df = st.session_state['merged_df'].copy()
        for col in required_columns:
            df[col] = df[col].where(pd.notna(df[col]), None)  # Convert NaN/None to JS null

        st.session_state['merged_df_processed'] = df        

        point_layer = pdk.Layer(
            'PointCloudLayer',
            data=st.session_state['merged_df_processed'].to_dict('records'),  # Convert to pydeck-compatible format
            get_position='[longitude, latitude]',
            get_units='meters',  # More precise than 'common'
            point_size=6,
            get_color=[238, 154, 0],  # Orange color
            pickable=True,
            # Add these properties for better tooltip handling
            auto_highlight=True,
            highlight_color=[255, 255, 0, 255],  # Yellow highlight
            radius_min_pixels=2,
            radius_max_pixels=10,
            # Explicitly include all tooltip properties
            _properties={col: col for col in required_columns}
        )

        # Create a copy of selected columns of merged_df in merged_df_str 
        st.session_state['merged_df_str'] = st.session_state['merged_df'][['longitude', 'latitude', st.session_state['data_var']]].copy()
        # Convert to string the numeric variables
        st.session_state['merged_df_str'][st.session_state['data_var']] = st.session_state['merged_df_str'][st.session_state['data_var']].astype('string')
        # Remove decimal part of some variables
        # Remove decimal part of some variables
        if st.session_state['data_var'] in ['HR', 'ww', 'N']:
            st.session_state['merged_df_str'][st.session_state['data_var']] = st.session_state['merged_df_str'][st.session_state['data_var']].str.split('.').str[0]

        # Define the text layer
        text_layer = pdk.Layer(
            'TextLayer',
            data=st.session_state['merged_df_str'].to_dict('records'),
            get_position='[longitude, latitude + 0.05]',
            get_text=st.session_state['data_var'],
            get_size=14,
            get_color=[0, 0, 0],
            background=True,                        # Enable background for text box
            background_color=[255, 255, 0, 200],    # RGBA color for background
            background_padding=5,                   # Padding around the text 
            get_angle=0,
            get_text_anchor='"middle"',
            get_alignment_baseline='"center"'
        )

        # Change the tooltip items according to the observation hour
        if st.session_state['selected_hour'] in [1, 7, 13, 19]:
            tooltip_html = """
            <div style="font-family: Arial, sans-serif; font-size: 12px;">
                <b>Indicativo:</b> {OMM} <br/>
                <b>Estación:</b> {Nombre} <br/>
                <b>Latitud:</b> {latitude} ° <br/>
                <b>Longitud:</b> {longitude} ° <br/>
                <b>Altura:</b> {altura} m <br/>
                <b>Temperatura:</b> {T} °C <br/>
                <b>Temp. Mín:</b> {Tn} °C <br/>
                <b>Temp. Máx:</b> {Tx} °C <br/>
                <b>Trocío:</b> {Td} °C <br/>
                <b>Humedad Relativa:</b> {HR} % <br/>
                <b>Índice Calor:</b> {IC} °C <br/>
                <b>Presión NMM:</b> {Pnmm} hPa <br/>
                <b>Velocidad:</b> {ff} km/h <br/>
                <b>Dirección:</b> {dd} <br/>
                <b>Lluvia 1(3) hora(s):</b> {R3} mm <br/>
                <b>Lluvia 6 horas:</b> {R6} mm <br/>
                <b>Lluvia 24 horas:</b> {R24} mm <br/>
                <b>Racha máxima:</b> {fx} km/h <br/>
                <b>Dirección:</b> {fxdd} <br/>
                <b>Nubosidad:</b> {N} octas <br/>
                <b>Tiempo:</b> {wwd}
            </div>
            """
        else:
            tooltip_html = """
            <div style="font-family: Arial, sans-serif; font-size: 12px;">
                <b>Indicativo:</b> {OMM} <br/>
                <b>Estación:</b> {Nombre} <br/>
                <b>Latitud:</b> {latitude} ° <br/>
                <b>Longitud:</b> {longitude} ° <br/>
                <b>Altura:</b> {altura} m <br/>
                <b>Temperatura:</b> {T} °C <br/>
                <b>Trocío:</b> {Td} °C <br/>
                <b>Humedad Relativa:</b> {HR} % <br/>
                <b>Índice Calor:</b> {IC} °C <br/>
                <b>Presión NMM:</b> {Pnmm} hPa <br/>
                <b>Velocidad:</b> {ff} km/h <br/>
                <b>Dirección:</b> {dd} <br/>
                <b>Lluvia 1(3) hora(s):</b> {R3} mm <br/>
                <b>Racha máxima:</b> {fx} km/h <br/>
                <b>Dirección:</b> {fxdd} <br/>
                <b>Nubosidad:</b> {N} octas <br/>
                <b>Tiempo:</b> {wwd}
            </div>
            """

        view_state = pdk.ViewState(
            latitude=21.5,
            longitude=-79.5,
            zoom=6.5,
            min_zoom=5.8,
            max_zoom=8,
            pitch=0,
        )

        deck = pdk.Deck(
            layers=[point_layer, text_layer],
            initial_view_state=view_state,
            tooltip={"html": tooltip_html, "style": {"backgroundColor": "yellow", "color": "black"}},
            map_style='light'
            #map_style='mapbox://styles/mapbox/light-v9'
        )

        st.pydeck_chart(deck)

        filtered_df = st.session_state['merged_df'][['OMM', 'Nombre', 'Prov', 'Reg', 'Fecha', st.session_state['data_var'], st.session_state['data_flag']]]

        # Define custom orders for each column
        province_order = ['IJU', 'PRI', 'ART', 'MAY', 'HAB', 'MTZ', 'VCL', 'CFG', 'SSP', 'CAV', 'CMG', 'LTU','HLG', 'GRA', 'SCU', 'GMO']
        region_order = ['OCC', 'CEN', 'OTE']
        filtered_df.loc[:, 'Prov'] = pd.Categorical(filtered_df['Prov'], categories=province_order, ordered=True)
        filtered_df.loc[:, 'Reg'] = pd.Categorical(filtered_df['Reg'], categories=region_order, ordered=True)

        st.session_state['sorted_df'] = (filtered_df.assign(
                Prov=lambda x: pd.Categorical(x['Prov'], categories=province_order, ordered=True),
                Reg=lambda x: pd.Categorical(x['Reg'], categories=region_order, ordered=True)
            )
            .sort_values(['Reg', 'Prov'])
        )
        
        with st.expander(f"Datos del Mapa de {st.session_state['data_name']}"):
            if st.session_state['data_var'] == 'dd':
                column_config = {
                    "OMM": st.column_config.TextColumn("OMM", width="small"),
                    "EMA": st.column_config.TextColumn("EMA", width="small"),
                    "Nombre": st.column_config.TextColumn("Nombre", width="medium"),
                    "Prov": st.column_config.TextColumn("Provincia", width="small"),
                    "Reg": st.column_config.TextColumn("Región", width="small"),
                    "Fecha": st.column_config.DatetimeColumn("Fecha", width="medium"),
                    st.session_state['data_var']: st.column_config.TextColumn(st.session_state['data_var'], width="small"),
                    st.session_state['data_flag']: st.column_config.NumberColumn(
                        st.session_state['data_flag'],
                        width="small"
                    )
                }
            elif st.session_state['data_var'] in ['HR', 'N']:
                column_config = {
                    "OMM": st.column_config.TextColumn("OMM", width="small"),
                    "EMA": st.column_config.TextColumn("EMA", width="small"),
                    "Nombre": st.column_config.TextColumn("Nombre", width="medium"),
                    "Prov": st.column_config.TextColumn("Provincia", width="small"),
                    "Reg": st.column_config.TextColumn("Región", width="small"),
                    "Fecha": st.column_config.DatetimeColumn("Fecha", width="medium"),
                    st.session_state['data_var']: st.column_config.NumberColumn(st.session_state['data_var'], width="small", format="%.0f"),
                    st.session_state['data_flag']: st.column_config.NumberColumn(
                        st.session_state['data_flag'],
                        width="small"
                    )
                }
            else:
                column_config = {
                    "OMM": st.column_config.TextColumn("OMM", width="small"),
                    "EMA": st.column_config.TextColumn("EMA", width="small"),
                    "Nombre": st.column_config.TextColumn("Nombre", width="medium"),
                    "Prov": st.column_config.TextColumn("Provincia", width="small"),
                    "Reg": st.column_config.TextColumn("Región", width="small"),
                    "Fecha": st.column_config.DatetimeColumn("Fecha", width="medium"),
                    st.session_state['data_var']: st.column_config.NumberColumn(st.session_state['data_var'], width="small", format="%.1f"),
                    st.session_state['data_flag']: st.column_config.NumberColumn(
                        st.session_state['data_flag'],
                        width="small"
                    )
                }
            st.data_editor(
                st.session_state['sorted_df'],
                column_order=('OMM', 'EMA', 'Nombre', 'Prov', 'Reg', 'Fecha', st.session_state['data_var'], st.session_state['data_flag']),
                column_config=column_config,
                hide_index=True,
                disabled=True,
                use_container_width=True
            )

        if (st.session_state['data_var'] != 'dd' and st.session_state['data_var'] != 'ww' and st.session_state['data_var'] != 'N'):
            with st.expander(f"Resumen Estadístico de los Datos del Mapa de {st.session_state['data_name']}"):
                stats_df = st.session_state['sorted_df'][[st.session_state['data_var']]].describe().T
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
 
                stats_df = stats_df.reset_index()

                st.data_editor(
                    stats_df,
                    column_config=column_config,
                    hide_index=True,
                    disabled=True
                )

                # Create the condition
                condition = (st.session_state['sorted_df'][st.session_state['data_flag']] > 4)
                # Apply the nulling only where condition is True
                st.session_state['sorted_df'].loc[condition, st.session_state['data_var']] = pd.NA
                # Show how many values were nulled
                nulled_count = condition.sum()

                if st.session_state['data_var'] in ['T', 'Tn', 'Tx', 'Pnmm', 'dP3', 'dP24', 'Td', 'HR', 'IC']:

                    report = statistical_report(
                        df=st.session_state['sorted_df'],
                        var=st.session_state['data_var'],
                        var_str=st.session_state['data_name'],
                        unit_str=st.session_state['data_unit'],
                        nulled_count_var=nulled_count
                    )
            
                elif st.session_state['data_var'] in ['R3', 'R6', 'R24', 'ff', 'fx']:
                    
                    report = statistical_report(
                                df=st.session_state['sorted_df'],
                                var=st.session_state['data_var'],
                                var_str=st.session_state['data_name'],
                                unit_str=st.session_state['data_unit'],
                                nulled_count_var=nulled_count,
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