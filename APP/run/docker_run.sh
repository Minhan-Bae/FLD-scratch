#!/bin/bash
docker build -t komedi-api .
docker run --name komedi-api -p 8000:8000 -p 80:8501 komedi-api