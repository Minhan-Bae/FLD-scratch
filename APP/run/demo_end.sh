#!/bin/bash

streamlit_pid=$(pgrep -f streamlit)
if [ -n "$streamlit_pid" ]; then
    kill -9 $streamlit_pid
fi