# Deployment Overview

## Cloud Deployment

The application was deployed on a GPU-enabled VM on Google Cloud Platform.

### Live Endpoint

http://34.106.163.171:8080/

### Command Used (on VM)

```bash
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

---

## Docker Support (Reproducibility)

Docker is included to enable consistent setup across environments.

### 1. Build Docker Image

```bash
docker build -t travel-app .
```

### 2. Run Docker Container

```bash
docker run -p 8080:8080 travel-app
```

### 3. Run in Background (Optional)

```bash
docker run -d -p 8080:8080 --name travel-app-container travel-app
```

### 4. Stop Container

```bash
docker stop travel-app-container
docker rm travel-app-container
```

### 5. Rebuild After Changes

```bash
docker build -t travel-app .
docker rm -f travel-app-container
docker run -d -p 8080:8080 --name travel-app-container travel-app
```