import streamlit as st
import pyodbc
import logging
from contextlib import contextmanager

@st.cache_resource(show_spinner='Conectandose al Servidor SQL...')
def get_connection():
    try:
        if not all(key in st.secrets.sql_server for key in ['server', 'username', 'password']):
            raise ValueError('Faltan parámetros de conexión en el fichero secrets.toml')
            
        conn_str = f"""
            DRIVER={st.secrets.sql_server.get('driver', 'ODBC Driver 17 for SQL Server')};
            SERVER={st.secrets.sql_server.server};
            DATABASE={st.secrets.sql_server.database};
            UID={st.secrets.sql_server.username};
            PWD={st.secrets.sql_server.password};
            Encrypt={st.secrets.sql_server.get('encrypt', 'no')};
            TrustServerCertificate={st.secrets.sql_server.get('trust_cert', 'no')};
        """
        return pyodbc.connect(conn_str)
    except Exception as e:
        st.error('Error de configuración - compruebe el fichero secrets.toml')
        logging.exception(e)
        return None

@contextmanager
def managed_database_connection():
    """Context manager for database connection"""
    conn = None
    try:
        conn = get_connection()
        yield conn
    except Exception as e:
        st.error(f"Error de conexión a la base de datos: {str(e)}")

@contextmanager
def database_cursor():
    """Context manager for safe cursor operations"""
    conn = get_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()  # Commit if no errors
    except Exception as e:
        conn.rollback()
        st.error(f"Error de la base de datos: {str(e)}")