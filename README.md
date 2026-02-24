# NeuroVistaFM

NeuroVistaFM is a real-time cognitive stability monitoring and adaptive prompting system for awake mapping workflows.

## Overview

The system includes:

- Doctor Console (Streamlit)
- Patient TV Display (Streamlit)
- Real-time Stress Instability Index (SII)
- Domain-level Functional Stability Index (FSI)
- Episode detection with hysteresis
- End-of-session SSI summary

## Run Instructions

Install dependencies:

pip install -r requirements.txt

Run doctor console:

streamlit run doctor_console.py --server.port 8501

Run TV display:

streamlit run tv_display.py --server.port 8502

Make sure both use the same Session ID.

## Notes

- MedGemma integration is stub-enabled (safe demo mode).
- No patient data is stored.
- Decision-support only. Not for surgical action.