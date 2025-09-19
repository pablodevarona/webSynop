# 🌦️ API de Observaciones Meteorológicas

Esta API proporciona observaciones meteorológicas de estaciones del INSMET para una fecha específica. Está construida con **FastAPI** y se conecta a una base de datos mediante **ODBC**.

## 🚀 Características

- Consulta de datos por fecha (`obs_time`)
- Información por estación: temperatura, humedad, presión, viento y lluvia
- Manejo de datos faltantes con `null`
- Documentación automática con Swagger UI (`/docs`)
- Configuración segura mediante archivo `.env`

## 📦 Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
