import streamlit as st
import plotly.express as px
import pandas as pd
from .database import database_cursor

# Function to convert degrees to compass directions
def degrees_to_direction(degrees, num_sectors=16):
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    sector_size = 360 / num_sectors
    try:
        index = int((degrees + sector_size/2) % 360 // sector_size)
        return directions[index]
    except Exception:
        return degrees
   
# Function to find out the max date-time from DB
def get_max_observation_time():
    try:
        with database_cursor() as cursor:
            cursor.execute("SELECT MAX(obs_time) AS max_datetime FROM Observations")
            result = cursor.fetchone()
            if result and result[0]:  # Check both result existence and value
                return result[0]
            st.warning("No se encontraron registros en la tabla Observations")
            return None
    except Exception as e:
        st.error(f"No se pudo hallar la fecha-hora de observación máxima: {str(e)}")
        return None
    
# Function to find out the max data_var value, station and date-time from DB
def find_max_data_var(df, data_var):
    try:
        if not all(col in df.columns for col in [data_var, 'Nombre', 'Fecha']):
            raise ValueError('Le faltan columnas requeridas al Data Frame')
        max_value = df[data_var].max()
        max_rows = df[df[data_var] == max_value]
        stations_dates = list(zip(max_rows['Nombre'], max_rows['Fecha']))
        return (max_value, stations_dates)
    except Exception as e:
        st.error(f"Error al buscar {data_var} máxima: {str(e)}")
        return (None, [])

# Function to find out the min data_var value, station and date-time from DB
def find_min_data_var(df, data_var):
    try:
        if not all(col in df.columns for col in [data_var, 'Nombre', 'Fecha']):
            raise ValueError('Le faltan columnas requeridas al Data Frame')
        min_value = df[data_var].min()
        min_rows = df[df[data_var] == min_value]
        stations_dates = list(zip(min_rows['Nombre'], min_rows['Fecha']))
        return (min_value, stations_dates)
    except Exception as e:
        st.error(f"Error al buscar {data_var} mínima: {str(e)}")
        return (None, [])
    
def statistical_report(df, var, var_str, unit_str, nulled_count_var, find_min=True, find_max=True):
    """
    Generate statistical report for a variable
    
    Args:
        df: DataFrame containing the data
        var: Column name to analyze
        var_str: Human-readable variable name
        unit_str: Measurement units
        find_min: Whether to calculate min values (default True)
        find_max: Whether to calculate max values (default True)
        
    Returns:
        dict: Contains nulled_info, min data (if calculated), and max data (if calculated)
    """
    report = {
        'nulled_info': None,
        'min': None,
        'max': None,
        'var_str': var_str,
        'unit_str': unit_str
    }
    
    # Handle nulled values count
    if nulled_count_var == 1:
        report['nulled_info'] = f'Se puso 1 valor de {var_str} en NULL basado en el banderín de calidad'
    elif nulled_count_var > 1:
        report['nulled_info'] = f'Se pusieron {nulled_count_var} valores de {var_str} en NULL basado en los banderines de calidad'
    
    # Get min data if requested
    if find_min:
        report['min'] = find_min_data_var(df, var)
    
    # Get max data if requested
    if find_max:
        report['max'] = find_max_data_var(df, var)
    
    return report

def generate_quality_report(quality_df):
    """
    Generate a quality control report for columns ending with 'q' (quality flags)
    
    Args:
        quality_df: DataFrame containing quality flag columns
        
    Returns:
        dict: Quality statistics for each variable
    """
    # Get all quality flag columns (ending with 'q')
    q_columns = [col for col in quality_df.columns if col.endswith('q')]
    
    report = {}
    
    for q_col in q_columns:
        # Base variable name (without 'q')
        base_var = q_col[:-1]
        
        # Calculate quality statistics
        null_count = quality_df[q_col].isnull().sum()
        good_count = (quality_df[q_col] < 3).sum()
        bad_count = ((quality_df[q_col] >= 3) & (quality_df[q_col].notnull())).sum()
        
        report[base_var] = {
            'quality_column': q_col,
            'total_observations': len(quality_df),
            'missing_flags': null_count,
            'good_quality': good_count,
            'bad_quality': bad_count,
            'good_percentage': (good_count / len(quality_df)) * 100 if len(quality_df) > 0 else 0,
            'bad_percentage': (bad_count / len(quality_df)) * 100 if len(quality_df) > 0 else 0
        }
    
    return report

def display_quality_report(report):
    """
    Display the quality report in Streamlit
    
    Args:
        report: Quality report dictionary from generate_quality_report()
    """
    st.markdown("<h3 style='font-size: 16px;'>Reporte de Calidad de Datos</h3>", unsafe_allow_html=True)
    
    for var, stats in report.items():
        with st.expander(f"Variable: {var}"):
            cols = st.columns(4)
            cols[0].metric("Observaciones Totales", stats['total_observations'])
            cols[1].metric("Datos Buenos", 
                          f"{stats['good_quality']}",
                          f"{stats['good_percentage']:.1f}%")
            cols[2].metric("Datos con Error", 
                          f"{stats['bad_quality']}",
                          f"{stats['bad_percentage']:.1f}%")
            cols[3].metric("Observaciones Faltantes", stats['missing_flags'])
            
            # Display quality distribution
            fig = px.pie(
                names=['Buenos', 'Errores', 'Faltantes'],
                values=[stats['good_quality'], stats['bad_quality'], stats['missing_flags']],
                title=f"Distribución de Calidad - {var}"
            )
            st.plotly_chart(fig, use_container_width=True)

def filter_6hour_intervals(df, date_col='Fecha', cols_to_keep=None):
    """
    Filters DataFrame to 6-hour intervals (00, 06, 12, 18)
    
    Args:
        df: Input DataFrame
        date_col: Name of datetime column
        cols_to_keep: List of columns to keep (None keeps all)
        
    Returns:
        Filtered DataFrame
    """
    default_cols = ['OMM','EMA','Nombre','Prov','Reg','Fecha','Tnq','Txq','Rs1q','R24q']
    cols = cols_to_keep if cols_to_keep else default_cols
    
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        return df[df[date_col].dt.hour % 6 == 0][cols]
    except Exception as e:
        st.error(f"Error filtering 6-hour intervals: {str(e)}")
        return pd.DataFrame(columns=cols)
    
import pandas as pd
import streamlit as st

def filter_specific_hours(df, hours, date_col='Fecha', cols_to_keep=None):
    """
    Filters DataFrame to specific hours of the day
    
    Args:
        df: Input DataFrame
        hours: List of hours to keep (e.g., [1, 7, 13, 19])
        date_col: Name of datetime column (default 'Fecha')
        cols_to_keep: List of columns to keep (None keeps all)
        
    Returns:
        Filtered DataFrame
    """
    default_cols = ['OMM','EMA','Nombre','Prov','Reg','Fecha','Tnq','Txq','Rs1q','R24q']
    cols = cols_to_keep if cols_to_keep else default_cols
    
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        return df[df[date_col].dt.hour.isin(hours)][cols]
    except Exception as e:
        st.error(f"Error filtering hours: {str(e)}")
        return pd.DataFrame(columns=cols)