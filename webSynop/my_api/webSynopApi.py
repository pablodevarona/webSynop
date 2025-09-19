from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import datetime
import pyodbc
from contextlib import contextmanager
import os
from dotenv import load_dotenv
import logging
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or allows specifics domains: allow_origins=["https://miapp.com", "https://admin.miapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load variables in file .env
load_dotenv()

# Configure logging
logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Input and Outpt Models

class ObservationRequest(BaseModel):
    obs_date: datetime.datetime = Field(..., description="Fecha de observación en formato ISO (ej. 2025-07-01T07:00:00)")

class Temperature(BaseModel):
    quality: Optional[str] = Field(None, description="Calidad del dato de temperatura")
    value: Optional[float] = Field(None, description="Valor de temperatura en grados Celsius")
    unit: str = Field(default="°C", description="Unidad de medida")    

class RelativeHumidity(BaseModel):
    quality: Optional[str] = Field(None, description="Calidad del dato de humedad relativa")
    value: Optional[float] = Field(None, description="Valor de humedad relativa en porcentaje")
    unit: str = Field(default="%", description="Unidad de medida")

class Pressure(BaseModel):
    quality: Optional[str] = Field(None, description="Calidad del dato de presión atmosférica al nivel medio del mar")
    value: Optional[float] = Field(None, description="Valor de presión atmosférica al nivel medio del mar en hPa")
    unit: str = Field(default="hPa", description="Unidad de medida")

class WindSpeed(BaseModel):
    quality: Optional[str] = Field(None, description="Calidad del dato de velocidad del viento")
    value: Optional[float] = Field(None, description="Velocidad del viento en km/h")
    unit: str = Field(default="km/h", description="Unidad de medida")

class WindDirection(BaseModel):
    value: Optional[str] = Field(None, description="Dirección cardinal del viento")
    calm: Optional[bool] = Field(None, description="Indica si el viento está en calma")    

class Wind(BaseModel):
    speed: WindSpeed
    direction: WindDirection

class RainAmount(BaseModel):
    value: Optional[float] = Field(None, description="Cantidad de lluvia en mm")
    quality: Optional[str] = Field(None, description="Calidad del dato de lluvia")

class Rain(BaseModel):
    rain_s3: RainAmount = Field(..., description="Lluvia acumulada en las últimas 1 o 3 horas")
    rain_24h: RainAmount = Field(..., description="Lluvia acumulada en las últimas 24 horas")
    unit: str = Field(default="mm", description="Unidad de medida para la lluvia")

class ObservationResponse(BaseModel):
    station_id: str
    station_name: str
    aws: bool
    obs_time: str
    temperature: Optional[Temperature]
    relative_humidity: Optional[RelativeHumidity]
    pressure: Optional[Pressure]
    wind: Optional[Wind]
    rain: Optional[Rain]

def format_obs_time(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()

def degrees_to_direction(degrees, num_sectors=16):
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    sector_size = 360 / num_sectors
    try:
        index = int((degrees + sector_size/2) % 360 // sector_size)
        return directions[index]
    except Exception:
        return degrees
    
def map_quality_description(code: Optional[int]) -> Optional[str]:
    if code is None:
        return None
    return {
        0: "Válido (sin control estadístico)",
        1: "Valor probable (entre los percentiles 0.3 y 99.7)",
        2: "Valor poco probable (menor que el percentil 0.3)",
        3: "Valor poco probable (mayor que el percentil 99.7)",
        4: "Error probable (menor o igual que el valor atípico)",
        5: "Error probable (mayor o igual que el valor atípico)",
        6: "Error instrumental",
        7: "Error de coherencia"
    }.get(code, "Desconocido")

def safe_quality(code):
    return map_quality_description(code) if pd.notna(code) else None

# Data Base Connection
def get_connection_fastapi():
    try:
        conn_str = (
            f"DRIVER={os.getenv('DB_DRIVER')};"
            f"SERVER={os.getenv('DB_SERVER')};"
            f"DATABASE={os.getenv('DB_DATABASE')};"
            f"UID={os.getenv('DB_USERNAME')};"
            f"PWD={os.getenv('DB_PASSWORD')};"
            f"Encrypt={os.getenv('DB_ENCRYPT')};"
            f"TrustServerCertificate={os.getenv('DB_TRUST_CERT')};"
        )
        return pyodbc.connect(conn_str)
    except Exception as e:
        logging.error("Error al conectar a la base de datos:")
        logging.error(traceback.format_exc())
        raise

@contextmanager
def managed_database_connection():
    conn = get_connection_fastapi()
    try:
        yield conn
    finally:
        conn.close()

# Endpoint principal
@app.post("/api/observation/", response_model=List[ObservationResponse])
def get_observations(request: ObservationRequest = Body(...)):
    query = """
        SELECT
               o.station_id,
               st.name AS station_name,
               o.aws,
               o.obs_time,
               o.air_temperature AS temperature,
               o.air_temperature_flag AS temperature_q,
               o.sea_level_pressure AS pressure,
               o.station_pressure_flag AS pressure_q,
               o.relative_humidity,
               o.relative_humidity_flag AS relative_humidity_q,
               o.precipitation_s3 AS rain_s3,
               o.precipitation_s3_flag AS rain_s3_q,
               o.precipitation_24h AS rain_24h,
               o.precipitation_24h_flag AS rain_24h_q,
               o.surface_wind_speed AS wind_speed,
               o.surface_wind_speed_flag AS wind_speed_q,
               o.surface_wind_direction AS wind_direction,
               o.surface_wind_direction_calm AS wind_calm
        FROM Observations o
        LEFT JOIN Stations st ON o.station_id = st.station_id
        WHERE o.obs_time = ?
        ORDER BY o.station_id
    """

    try:
        with managed_database_connection() as conn:
            df = pd.read_sql(query, conn, params=[request.obs_date])

        if df.empty:
            raise HTTPException(status_code=404, detail="No se encontraron observaciones para esa fecha")

        # substituting the data frame NaN values (not valid in Jason) with None
        df = df.astype(object).where(pd.notna(df), None)

        # degrees to direction conversion
        numeric_mask = df['wind_direction'].apply(lambda x: isinstance(x, (int, float)))
        df.loc[numeric_mask, 'wind_direction'] = df.loc[numeric_mask, 'wind_direction'].apply(degrees_to_direction)

        # m/s to km/h conversion
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
        # Multiplicar solo si el valor es numérico
        df["wind_speed"] = (df["wind_speed"] * 3.6).round(1)
        
        response = []

        for _, row in df.iterrows():
            try:

                obs = ObservationResponse(
                    station_id=row["station_id"],
                    station_name=row["station_name"],
                    aws=row["aws"],
                    obs_time=format_obs_time(row["obs_time"]),

                    temperature=Temperature(
                        quality=safe_quality(row["temperature_q"]),
                        value=row["temperature"],
                        unit="°C"
                    ),

                    relative_humidity=RelativeHumidity(
                        quality=safe_quality(row["relative_humidity_q"]),
                        value=row["relative_humidity"],
                        unit="%"
                    ),
                    pressure=Pressure(
                        quality=safe_quality(row["pressure_q"]),
                        value=row["pressure"],
                        unit="hPa"
                    ),

                    wind = Wind(
                        speed=WindSpeed(
                            quality=safe_quality(row["wind_speed_q"]),
                            value=row["wind_speed"],
                            unit="km/h"
                        ),
                        direction=WindDirection(
                            value=row["wind_direction"],
                            calm=row["wind_calm"]
                        )
                    ),

                    rain = Rain(
                        rain_s3=RainAmount(
                            value=row["rain_s3"],
                            quality=safe_quality(row["rain_s3_q"])
                        ),
                        rain_24h=RainAmount(
                            value=row["rain_24h"],
                            quality=safe_quality(row["rain_24h_q"])
                        ),
                        unit="mm"
                    )

                )

                response.append(obs)

            except Exception as row_error:
                logging.error("Error al construir la respuesta para una fila:")
                logging.error(traceback.format_exc())
                continue  # omite esta fila y sigue con las demás

        return response

    except Exception as e:
        logging.error("Error general en el endpoint:")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error interno al procesar la solicitud")