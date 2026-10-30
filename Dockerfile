# ============================================================
# AR-MCDF  |  Dockerfile
# ============================================================
# Stage 1: Build React frontend
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ============================================================
# Stage 2: Python production image
# ============================================================
FROM python:3.11-slim

# System dependencies for PyTorch Geometric and torch_geometric
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy project files
COPY src/          ./src/
COPY dashboard/    ./dashboard/
COPY data/         ./data/

# Copy built frontend into Flask static folder so it's served by Gunicorn
COPY --from=frontend-build /app/frontend/dist/ ./dashboard/static/

# Copy trained models (must exist before building image)
# If models/ doesn't exist, the server will warn but still boot
COPY models/       ./models/

# Expose port
EXPOSE 8050

# Environment defaults
ENV FLASK_ENV=production
ENV ALLOWED_ORIGIN=http://localhost:8050

# Run with Gunicorn: 2 workers, bind to 0.0.0.0:8050
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8050", "--timeout", "120", "dashboard.server:app"]
