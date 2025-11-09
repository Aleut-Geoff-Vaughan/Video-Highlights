#!/bin/bash
# Launcher script for Video Highlights Web App

echo "Starting Video Highlights Web Interface..."
echo "Open your browser to the URL shown below"
echo ""
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
