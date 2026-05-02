#!/bin/bash

# ==========================================
# GCP DEPLOYMENT COMMAND LOG (REFERENCE)
# ==========================================
# This script documents the exact steps used
# to deploy the app on a GCP VM.
# It is not fully automated by design.
# ==========================================

# ---------- 1. SSH INTO VM ----------
# gcloud compute ssh <INSTANCE_NAME> --zone=<ZONE>

# ---------- 2. UPDATE SYSTEM ----------
sudo apt update -y

# ---------- 3. INSTALL PYTHON ----------
sudo apt install -y python3-pip

# ---------- 4. CLONE REPO ----------
# Replace with your repo
git clone <REPO_URL>
cd <REPO_NAME>

# ---------- 5. INSTALL DEPENDENCIES ----------
pip3 install -r requirements.txt

# ---------- 6. VERIFY GPU ----------
nvidia-smi

# ---------- 7. RUN APPLICATION ----------
streamlit run app.py \
  --server.port 8080 \
  --server.address 0.0.0.0

# ---------- 8. OPTIONAL: RUN IN BACKGROUND ----------
# nohup streamlit run app.py \
#   --server.port 8080 \
#   --server.address 0.0.0.0 \
#   > app.log 2>&1 &

# ==========================================
# ACCESS APP:
# http://<EXTERNAL_IP>:8080
# ==========================================