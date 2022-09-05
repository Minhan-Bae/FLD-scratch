#!/bin/bash

streamlit_pid=$(pgrep -f streamlit)
if [ -z "$streamlit_pid" ]; then
    docker exec komedi-api streamlit run ./fronted/front.py
fi