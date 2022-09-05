#!/bin/bash

docker stop komedi-api
docker rm -f komedi-api
docker rmi -f komedi-api