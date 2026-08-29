# ============================================================
# Shortcut operasional pipeline.
# Windows: pakai `make` dari Git Bash / WSL, atau salin perintah
# di kolom kanan langsung ke PowerShell.
# ============================================================

COMPOSE := docker compose
EXEC    := $(COMPOSE) exec -T pipeline

.DEFAULT_GOAL := help
.PHONY: help build up down restart ps logs verify pipeline phase1 phase2 phase3 phase4 shell spark-ui minio-ui dashboard clean nuke

help:  ## Tampilkan daftar perintah
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build:  ## Bangun image dasar (java + pyspark + konektor S3A)
	$(COMPOSE) build

up:  ## Jalankan seluruh stack di background
	$(COMPOSE) up -d

down:  ## Hentikan stack (volume data tetap aman)
	$(COMPOSE) down

restart: down up  ## Restart stack

ps:  ## Status seluruh service
	$(COMPOSE) ps

logs:  ## Ikuti log semua service
	$(COMPOSE) logs -f --tail=100

verify:  ## Smoke test environment (MinIO + Spark + S3A)
	$(EXEC) python scripts/verify_env.py

pipeline:  ## Jalankan seluruh pipeline Fase 1-4
	$(EXEC) python scripts/run_pipeline.py

phase1:  ## Fase 1 - ingestion CSV ke MinIO
	$(EXEC) python -m src.phases.phase1_collect

phase2:  ## Fase 2 - cleaning & text preprocessing
	$(EXEC) python -m src.phases.phase2_clean

phase3:  ## Fase 3 - EDA Spark SQL
	$(EXEC) python -m src.phases.phase3_explore

phase4:  ## Fase 4 - training model multi-head
	$(EXEC) python -m src.phases.phase4_model

dashboard:  ## URL dashboard Streamlit (Fase 5)
	@echo "http://localhost:8501"

shell:  ## Masuk ke container driver
	$(COMPOSE) exec pipeline bash

spark-ui:  ## URL Spark Master UI
	@echo "http://localhost:8080"

minio-ui:  ## URL MinIO Console
	@echo "http://localhost:9001"

clean:  ## Hapus container + network (volume MinIO tetap)
	$(COMPOSE) down --remove-orphans

nuke:  ## Hapus SEMUANYA termasuk data lake di volume MinIO
	$(COMPOSE) down -v --remove-orphans
