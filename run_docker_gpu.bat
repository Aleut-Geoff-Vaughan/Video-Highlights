@echo off
docker compose --profile gpu up --build api worker-gpu api-client processing-ui admin-global admin-tenant
