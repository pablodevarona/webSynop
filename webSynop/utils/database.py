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
    
@st.cache_data(ttl=600)
def run_query(query, *params):
    with get_connection().cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()