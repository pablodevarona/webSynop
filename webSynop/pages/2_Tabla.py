import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import sys
import ast
from utils.shared_functions import degrees_to_direction
from utils.shared_functions import get_max_observation_time
from utils.database import managed_database_connection
from utils.shared_functions import statistical_report
from utils.shared_functions import generate_quality_report, display_quality_report
from utils.shared_functions import filter_specific_hours

st.set_page_config(layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; font-size: 24px; color: blue;'>
        Observaciones de Superficie
    </h1>
    """,
    unsafe_allow_html=True
)

selected_hours = [1, 4, 7, 10, 13, 16, 19, 22]

stationDic_0 = {}

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
st.session_state['selected_period'] = st.sidebar.selectbox('Seleccione el Período', ['1 hora', '08:00-19:00', '20:00-07:00', '24 horas'], index=3)

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
        Observations.aws,
        Observations.air_temperature,
        Observations.air_temperature_flag,
        Observations.minimum_temperature,
        Observations.minimum_temperature_flag,
        Observations.maximum_temperature,
        Observations.maximum_temperature_flag,
        Observations.station_pressure,
        Observations.station_pressure_flag,
        Observations.sea_level_pressure,
        Observations.pressure_tendency,
        Observations.pressure_change_3h,
        Observations.pressure_change_24h,
        Observations.dewpoint_temperature,
        Observations.dewpoint_temperature_flag,
        Observations.relative_humidity,
        Observations.relative_humidity_flag,
        Observations.saturation_deficit,
        Observations.heat_index,
        Observations.precipitation_s3,
        Observations.precipitation_s3_trace,
        Observations.precipitation_s3_flag,
        Observations.precipitation_s3_period,
        Observations.precipitation_s1,
        Observations.precipitation_s1_trace,
        Observations.precipitation_s1_flag,
        Observations.precipitation_24h,
        Observations.precipitation_24h_trace,
        Observations.precipitation_24h_flag,
        Observations.surface_wind_speed,
        Observations.surface_wind_speed_flag,
        Observations.surface_wind_direction_calm,
        Observations.surface_wind_direction,
        Observations.present_weather,
        Observations.past_weather_1,
        Observations.past_weather_2,
        Observations.highest_gust_speed,
        Observations.highest_gust_speed_flag,
        Observations.highest_gust_direction,
        Observations.highest_gust_date,
        Observations.temperature_change,
        Observations.temperature_change_flag,
        Observations.temperature_change_date,
        Observations.evapotranspiration,
        Observations.evapotranspiration_flag,
        Observations.evapotranspiration_type,
        Observations.sunshine,
        Observations.sunshine_flag,
        Observations.sunshine_period,
        Observations.global_solar_radiation,
        Observations.global_solar_radiation_flag,
        Observations.global_solar_radiation_period,
        Observations.ground_state,
        Observations.horizontal_visibility,
        Observations.cloud_cover,
        Observations.cloud_cover_obscured,
        Observations.low_cloud_amount,
        Observations.low_cloud_type,
        Observations.middle_cloud_type,
        Observations.high_cloud_type,
        Observations.lowest_cloud_base_min,
        Observations.lowest_cloud_base_max,
        Observations.tropical_sky_state,
        Observations.low_cloud_drift,
        Observations.middle_cloud_drift,
        Observations.high_cloud_drift,
        Observations.vertical_cloud_genus,
        Observations.vertical_cloud_direction,
        Observations.vertical_cloud_top,
        Observations.cloud_genus_layer_1,
        Observations.cloud_cover_layer_1,
        Observations.cloud_height_layer_1,
        Observations.cloud_genus_layer_2,
        Observations.cloud_cover_layer_2,
        Observations.cloud_height_layer_2,
        Observations.cloud_genus_layer_3,
        Observations.cloud_cover_layer_3,
        Observations.cloud_height_layer_3,
        Observations.cloud_genus_layer_4,
        Observations.cloud_cover_layer_4,
        Observations.cloud_height_layer_4,
        Observations.sea_state,
        Observations.wind_speed,
        Observations.geopotential_surface,
        Observations.geopotential_height
FROM Observations
INNER JOIN Stations ON Observations.station_id = Stations.station_id
WHERE Observations.obs_time >= ? AND Observations.obs_time <= ?
ORDER BY Observations.station_id, Observations.obs_time DESC
"""
if st.button('Cargar los Datos'):
    try:
        with managed_database_connection() as conn:
            st.session_state['filtered_df'] = pd.read_sql(
                query, 
                conn, 
                params=[st.session_state['start_date'], st.session_state['end_date']]
            )

        if 'filtered_df' in st.session_state:
            # Express wind speed in km/h
            if 'surface_wind_speed' in st.session_state['filtered_df']:
                if pd.api.types.is_numeric_dtype(st.session_state['filtered_df']['surface_wind_speed']):
                    st.session_state['filtered_df'].loc[:, 'surface_wind_speed'] = (
                        st.session_state['filtered_df'].loc[:, 'surface_wind_speed'] * 3.6
                    ).round(1)
            if 'highest_gust_speed' in st.session_state['filtered_df']:
                if pd.api.types.is_numeric_dtype(st.session_state['filtered_df']['highest_gust_speed']):
                    st.session_state['filtered_df'].loc[:, 'highest_gust_speed'] = (
                        st.session_state['filtered_df'].loc[:, 'highest_gust_speed'] * 3.6
                    ).round(1)
            # Express wind direction in course
            if 'surface_wind_direction' in st.session_state['filtered_df']:
                numeric_mask = st.session_state['filtered_df']['surface_wind_direction'].apply(lambda x: isinstance(x, (int, float)))
                st.session_state['filtered_df'].loc[numeric_mask, 'surface_wind_direction'] = st.session_state['filtered_df'].loc[numeric_mask, 'surface_wind_direction'].apply(degrees_to_direction)
            if 'highest_gust_direction' in st.session_state['filtered_df']:
                numeric_mask = st.session_state['filtered_df']['highest_gust_direction'].apply(lambda x: isinstance(x, (int, float)))
                st.session_state['filtered_df'].loc[numeric_mask, 'highest_gust_direction'] = st.session_state['filtered_df'].loc[numeric_mask, 'highest_gust_direction'].apply(degrees_to_direction)        
            # Conditional sign flip (only if all required columns exist and are numeric and pressure_tendency > 5)
            df = st.session_state['filtered_df']
            cols_required = ['pressure_tendency', 'pressure_change_3h']

        st.session_state['filtered_df'].columns = [
                            'OMM',
                            'Nombre',
                            'Prov',
                            'Reg',
                            'Fecha',
                            'EMA',
                            'T',
                            'Tq',
                            'Tn',
                            'Tnq',
                            'Tx',
                            'Txq',
                            'P0',
                            'P0q',
                            'Pnmm',
                            'aP3',
                            'dP3',
                            'dP24',
                            'Td',
                            'Tdq',
                            'HR',
                            'HRq',
                            'dSat',
                            'IC',
                            'Rs3',
                            'Rs3T',
                            'Rs3q',
                            'Rs3P',
                            'Rs1',
                            'Rs1T',
                            'Rs1q',
                            'R24',
                            'R24T',
                            'R24q',
                            'ff',
                            'ffq',
                            'calma',
                            'dd',
                            'ww',
                            'W1',
                            'W2',
                            'fx',
                            'fxq',
                            'fxdd',
                            'fxf',
                            'dT',
                            'dTq',
                            'dTf',
                            'EEE',
                            'EEEq',
                            'EEEc',
                            'SSS',
                            'SSSq',
                            'SSSp',
                            'Rad',
                            'Radq',
                            'Radp',
                            'E',
                            'VV',
                            'N',
                            'Inv',
                            'Nh',
                            'CL',
                            'CM',
                            'CH',
                            'hmin',
                            'hmax',
                            'Cs',
                            'DL',
                            'DM',
                            'DH',
                            'C',
                            'Da',
                            'eC',
                            'C1',
                            'N1',
                            'hShS1',
                            'C2',
                            'N2',
                            'hShS2',
                            'C3',
                            'N3',
                            'hShS3',
                            'C4',
                            'N4',
                            'hShS4',
                            'S',
                            'Fx',
                            'a3',
                            'hhh'
                            ]
        
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

        st.session_state['temperature_df'] = st.session_state['filtered_df'][['OMM','EMA','Nombre','Prov','Reg','Fecha','T', 'Tq', 'Tn', 'Tnq', 'Tx', 'Txq','Td','Tdq','HR','HRq','dSat','IC','dT','dTf']]

        if 'temperature_df' in st.session_state:

            with st.expander('Tabla de Temperaturas (°C), Humedad Relativa (%), Déficit de saturación (hPa) e IC (°C)'):
                 st.data_editor(st.session_state['temperature_df'], column_order=('OMM','EMA','Nombre','Prov','Reg','Fecha','T', 'Tn', 'Tx', 'Td','HR','dSat','IC','dT','dTf'), hide_index=True, disabled=True)

            with st.expander('Resumen Estadístico de la Tabla de Temperaturas y Humedad'):
                stats_df = st.session_state['temperature_df'][['T','Tn','Tx','Td','HR','dSat','IC']].describe().T
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
                condition = (st.session_state['temperature_df']['Tq'] > 4)
                # Apply the nulling only where condition is True
                st.session_state['temperature_df'].loc[condition, 'T'] = pd.NA
                # Show how many values were nulled
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['temperature_df'],
                    var='T',
                    var_str='Temperatura',
                    unit_str='°C',
                    nulled_count_var=nulled_count
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

                condition = (st.session_state['temperature_df']['Tnq'] > 4)
                st.session_state['temperature_df'].loc[condition, 'Tn'] = pd.NA
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['temperature_df'],
                    var='Tn',
                    var_str='Temperatura Mínima',
                    unit_str='°C',
                    nulled_count_var=nulled_count
                )

                if report['nulled_info']:
                    st.info(report['nulled_info'])

                if report['min'] is not None:
                    min_value, min_stations_dates = report['min']
                    if min_value is not None and not np.isnan(min_value):
                        st.write(f"Mínimo de {report['var_str']}: {min_value} {report['unit_str']} observado en:")
                        for station, date in min_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                condition = (st.session_state['temperature_df']['Txq'] > 4)
                st.session_state['temperature_df'].loc[condition, 'Tx'] = pd.NA
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['temperature_df'],
                    var='Tx',
                    var_str='Temperatura Máxima',
                    unit_str='°C',
                    nulled_count_var=nulled_count
                )

                if report['nulled_info']:
                    st.info(report['nulled_info'])

                if report['min'] is not None:
                    min_value, min_stations_dates = report['min']
                    if min_value is not None and not np.isnan(min_value):
                        st.write(f"Mínimo de {report['var_str']}: {min_value} {report['unit_str']} observado en:")
                        for station, date in min_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                condition = (st.session_state['temperature_df']['Tdq'] > 4)
                st.session_state['temperature_df'].loc[condition, 'Td'] = pd.NA
                nulled_count_Td = condition.sum()

                report = statistical_report(
                    df=st.session_state['temperature_df'],
                    var='Td',
                    var_str='Temperatura de Punto de Rocío',
                    unit_str='°C',
                    nulled_count_var=nulled_count_Td
                )

                if report['nulled_info']:
                    st.info(report['nulled_info'])

                if report['min'] is not None:
                    min_value, min_stations_dates = report['min']
                    if min_value is not None and not np.isnan(min_value):
                        st.write(f"Mínimo de {report['var_str']}: {min_value} {report['unit_str']} observado en:")
                        for station, date in min_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                condition = (st.session_state['temperature_df']['HRq'] > 4)
                st.session_state['temperature_df'].loc[condition, 'HR'] = pd.NA
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['temperature_df'],
                    var='HR',
                    var_str='Humedad Relativa',
                    unit_str='%',
                    nulled_count_var=nulled_count
                )

                if report['nulled_info']:
                    st.info(report['nulled_info'])

                if report['min'] is not None:
                    min_value, min_stations_dates = report['min']
                    if min_value is not None and not np.isnan(min_value):
                        st.write(f"Mínimo de {report['var_str']}: {min_value} {report['unit_str']} observado en:")
                        for station, date in min_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                condition = (st.session_state['temperature_df']['Tq'] > 4) | (st.session_state['temperature_df']['HRq'] > 4)
                st.session_state['temperature_df'].loc[condition, 'dSat'] = pd.NA
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['temperature_df'],
                    var='dSat',
                    var_str='Déficit de Saturación',
                    unit_str='hPa',
                    nulled_count_var=nulled_count
                )

                if report['nulled_info']:
                    st.info(report['nulled_info'])

                if report['min'] is not None:
                    min_value, min_stations_dates = report['min']
                    if min_value is not None and not np.isnan(min_value):
                        st.write(f"Mínimo de {report['var_str']}: {min_value} {report['unit_str']} observado en:")
                        for station, date in min_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                condition = (st.session_state['temperature_df']['Tq'] > 4) | (st.session_state['temperature_df']['HRq'] > 4)
                st.session_state['temperature_df'].loc[condition, 'IC'] = pd.NA
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['temperature_df'],
                    var='IC',
                    var_str='Índice de Calor',
                    unit_str='°C',
                    nulled_count_var=nulled_count
                )

                if report['nulled_info']:
                    st.info(report['nulled_info'])

                if report['min'] is not None:
                    min_value, min_stations_dates = report['min']
                    if min_value is not None and not np.isnan(min_value):
                        st.write(f"Mínimo de {report['var_str']}: {min_value} {report['unit_str']} observado en:")
                        for station, date in min_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

        st.session_state['rain_df'] = st.session_state['filtered_df'][['OMM','EMA','Nombre','Prov','Reg','Fecha','Rs3','Rs3T','Rs3q','Rs3P','Rs1','Rs1T','Rs1q','R24','R24T','R24q','EEE','EEEq']]

        if 'rain_df' in st.session_state:

            with st.expander('Tabla de Precipitación (mm) y Evaporación (mm)'):
                st.data_editor(st.session_state['rain_df'], column_order=('OMM','EMA','Nombre','Prov','Reg','Fecha','Rs3','Rs3T','Rs3P','Rs1','Rs1T','R24','R24T','EEE'), hide_index=True, disabled=True)
            with st.expander('Resumen Estadístico de la Tabla de Precipitación (mm) y Evaporación (mm)'):
                stats_df = st.session_state['rain_df'][['Rs3','Rs1','R24','EEE']].describe().T
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
                condition = (st.session_state['rain_df']['Rs3q'] > 4)
                # Apply the nulling only where condition is True
                st.session_state['rain_df'].loc[condition, 'Rs3'] = pd.NA
                # Show how many values were nulled
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['rain_df'],
                    var='Rs3',
                    var_str='Lluvia en 1(3) hora(s)',
                    unit_str='mm',
                    nulled_count_var=nulled_count,
                    find_min=False
                )

                if report['nulled_info']:
                    st.info(report['nulled_info'])

                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                condition = (st.session_state['rain_df']['Rs1q'] > 4)
                st.session_state['rain_df'].loc[condition, 'Rs1'] = pd.NA
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['rain_df'],
                    var='Rs1',
                    var_str='Lluvia en 6 horas',
                    unit_str='mm',
                    nulled_count_var=nulled_count,
                    find_min=False
                )

                if report['nulled_info']:
                    st.info(report['nulled_info'])

                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                condition = (st.session_state['rain_df']['R24q'] > 4)
                st.session_state['rain_df'].loc[condition, 'R24'] = pd.NA
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['rain_df'],
                    var='R24',
                    var_str='Lluvia en 24 horas',
                    unit_str='mm',
                    nulled_count_var=nulled_count,
                    find_min=False
                )

                if report['nulled_info']:
                    st.info(report['nulled_info'])

                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                condition = (st.session_state['rain_df']['EEEq'] > 4)
                st.session_state['rain_df'].loc[condition, 'EEE'] = pd.NA
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['rain_df'],
                    var='EEE',
                    var_str='Evaporación',
                    unit_str='mm',
                    nulled_count_var=nulled_count,
                    find_min=False
                )

                if report['nulled_info']:
                    st.info(report['nulled_info'])

                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

        st.session_state['pressure_df'] = st.session_state['filtered_df'][['OMM','EMA','Nombre','Prov','Reg','Fecha','P0','P0q','Pnmm','hhh','aP3','dP3','dP24']]

        if 'pressure_df' in st.session_state:

            # Conditional sign flip (only where aP3 > 5)
            mask = st.session_state['pressure_df']['aP3'] > 5
            st.session_state['pressure_df'].loc[mask, 'dP3'] *= -1

            with st.expander('Tabla de Presión (hPa)'):
                 st.data_editor(st.session_state['pressure_df'], column_order=('OMM','EMA','Nombre','Prov','Reg','Fecha','P0','Pnmm','hhh','aP3','dP3','dP24'), hide_index=True, disabled=True)
            with st.expander('Resumen Estadístico de la Tabla de Presión'):
                stats_df = st.session_state['pressure_df'][['Pnmm','dP3','dP24']].describe().T
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
                condition = (st.session_state['pressure_df']['P0q'] > 4)
                # Apply the nulling only where condition is True
                st.session_state['pressure_df'].loc[condition, 'Pnmm'] = pd.NA
                # Show how many values were nulled
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['pressure_df'],
                    var='Pnmm',
                    var_str='Presión al nivel medio del mar',
                    unit_str='hPa',
                    nulled_count_var=nulled_count
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

                # Create the condition
                condition = (st.session_state['pressure_df']['P0q'] > 4)
                # Apply the nulling only where condition is True
                st.session_state['pressure_df'].loc[condition, 'dP3'] = pd.NA
                # Show how many values were nulled
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['pressure_df'],
                    var='dP3',
                    var_str='Cambio de Presión en 3 Horas',
                    unit_str='hPa',
                    nulled_count_var=nulled_count
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

                # Create the condition
                condition = (st.session_state['pressure_df']['P0q'] > 4)
                # Apply the nulling only where condition is True
                st.session_state['pressure_df'].loc[condition, 'dP24'] = pd.NA
                # Show how many values were nulled
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['pressure_df'],
                    var='dP24',
                    var_str='Cambio de Presión en 24 Horas',
                    unit_str='hPa',
                    nulled_count_var=nulled_count
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

        st.session_state['wind_df'] = st.session_state['filtered_df'][['OMM','EMA','Nombre','Prov','Reg','Fecha','ff','ffq','calma','dd','fx','fxq','fxdd','fxf']]

        if 'wind_df' in st.session_state:

            with st.expander('Tabla de Velocidad (km/h) y Dirección del Viento (rumbos)'):
                 st.data_editor(st.session_state['wind_df'], column_order=('OMM','EMA','Nombre','Prov','Reg','Fecha','ff','calma','dd','fx','fxdd','fxf'), hide_index=True, disabled=True)
            with st.expander('Resumen Estadístico de la Tabla de Velocidad y Dirección del Viento'):
                stats_df = st.session_state['wind_df'][['ff','fx']].describe().T
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
                condition = (st.session_state['wind_df']['ffq'] > 4)
                # Apply the nulling only where condition is True
                st.session_state['wind_df'].loc[condition, 'ff'] = pd.NA
                # Show how many values were nulled
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['wind_df'],
                    var='ff',
                    var_str='Velocidad del viento',
                    unit_str='km/h',
                    nulled_count_var=nulled_count,
                    find_min=False
                )

                # Display nulled values info if exists
                if report['nulled_info']:
                    st.info(report['nulled_info'])

                # Display max values if calculated
                if report['max'] is not None and not np.isnan(max_value):
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

                # Create the condition
                condition = (st.session_state['wind_df']['fxq'] > 4)
                # Apply the nulling only where condition is True
                st.session_state['wind_df'].loc[condition, 'fx'] = pd.NA
                # Show how many values were nulled
                nulled_count = condition.sum()

                report = statistical_report(
                    df=st.session_state['wind_df'],
                    var='fx',
                    var_str='Racha máxima del viento',
                    unit_str='km/h',
                    nulled_count_var=nulled_count,
                    find_min=False
                )

                # Display nulled values info if exists
                if report['nulled_info']:
                    st.info(report['nulled_info'])

                # Display max values if calculated
                if report['max'] is not None:
                    max_value, max_stations_dates = report['max']
                    if max_value is not None and not np.isnan(max_value):
                        st.write(f"Máximo de {report['var_str']}: {max_value} {report['unit_str']} observado en:")
                        for station, date in max_stations_dates:
                            st.write(f" - {station} el {date.strftime('%Y-%m-%d %H:%M')}")

        st.session_state['ww_df'] = st.session_state['filtered_df'][['OMM','EMA','Nombre','Prov','Reg','Fecha','VV','ww','W1','W2','E','S','Fx']]

        if 'ww_df' in st.session_state:

            with st.expander('Tabla de Visibilidad (m), Estado del Tiempo, del Suelo y del Mar (código)'):
                 st.data_editor(st.session_state['ww_df'], column_order=('OMM','EMA','Nombre','Prov','Reg','Fecha','VV','ww','W1','W2','E','S','Fx'), hide_index=True, disabled=True)

        st.session_state['cloud_df'] = st.session_state['filtered_df'][['OMM','EMA','Nombre','Prov','Reg','Fecha','N','Inv','Nh','CL','CM','CH','hmin','hmax','Cs','DL','DM','DH','C','Da','eC','C1','N1','hShS1','C2','N2','hShS2','C3','N3','hShS3','C4','N4','hShS4']]

        if 'cloud_df' in st.session_state:

            with st.expander('Tabla de Nubosidad: N (octas), D (rumbos), h (m) (código)'):
                 st.data_editor(st.session_state['cloud_df'], column_order=('OMM','EMA','Nombre','Prov','Reg','Fecha','N','Inv','Nh','CL','CM','CH','hmin','hmax','Cs','DL','DM','DH','C','Da','eC','C1','N1','hShS1','C2','N2','hShS2','C3','N3','hShS3','C4','N4','hShS4'), hide_index=True, disabled=True)

        st.session_state['quality_df_0'] = st.session_state['filtered_df'][['OMM','EMA','Nombre','Prov','Reg','Fecha','Tq','Tdq','HRq','Rs3q','P0q','ffq','fxq']]

        if 'quality_df_0' in st.session_state:
            with st.expander('Tabla de Banderines de Calidad'):
                 st.data_editor(st.session_state['quality_df_0'], column_order=('OMM','EMA','Nombre','Prov','Reg','Fecha','Tq','Tdq','HRq','Rs3q','P0q','ffq','fxq'), hide_index=True, disabled=True)

            quality_report = generate_quality_report(st.session_state['quality_df_0'])
            display_quality_report(quality_report)
        else:
            st.warning("No se encontró el DataFrame de calidad en session_state")

        st.session_state['quality_df_1'] = st.session_state['filtered_df'][['OMM','EMA','Nombre','Prov','Reg','Fecha','Tnq','Txq','Rs1q','R24q']]

        if 'quality_df_1' in st.session_state:
            # Apply filter
            st.session_state['filtered_SM_hours_df'] = filter_specific_hours(
                df=st.session_state['quality_df_1'],
                hours=[1, 7, 13, 19],
                date_col='Fecha',
                cols_to_keep=['OMM','EMA','Nombre','Prov','Reg','Fecha','Tnq','Txq','Rs1q','R24q']
            )

            with st.expander('Tabla de Banderines de Calidad (variables de observaciones SM)'):
                 st.data_editor(st.session_state['filtered_SM_hours_df'], column_order=('OMM','EMA','Nombre','Prov','Reg','Fecha','Tnq','Txq','Rs1q','R24q'), hide_index=True, disabled=True)

            quality_report_1 = generate_quality_report(st.session_state['filtered_SM_hours_df'])
            display_quality_report(quality_report_1)
        else:
            st.warning("No se encontró el DataFrame de calidad en session_state")

    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")