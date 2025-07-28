"""
Production Deployment Configuration

This module provides Docker, cloud deployment, and API service configurations.
"""

# Docker Configuration
DOCKERFILE_CONTENT = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed models outputs

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/health || exit 1

# Run the application
CMD ["streamlit", "run", "dashboard/simple_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

# Docker Compose for full stack
DOCKER_COMPOSE_CONTENT = """
version: '3.8'

services:
  ml-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:password@postgres:5432/stockdb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: stockdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ml-app

volumes:
  postgres_data:
  redis_data:
"""

# Kubernetes Deployment
KUBERNETES_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stock-predictor
  labels:
    app: stock-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stock-predictor
  template:
    metadata:
      labels:
        app: stock-predictor
    spec:
      containers:
      - name: stock-predictor
        image: your-registry/stock-predictor:latest
        ports:
        - containerPort: 8501
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: stock-predictor-service
spec:
  selector:
    app: stock-predictor
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8501
  type: LoadBalancer
"""

# Terraform for AWS Infrastructure
TERRAFORM_AWS = """
# Provider configuration
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  default     = "production"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "stock-predictor-vpc"
    Environment = var.environment
  }
}

# Subnets
resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name        = "public-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "stock-predictor-igw"
    Environment = var.environment
  }
}

# Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "public-route-table"
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "stock-predictor-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Environment = var.environment
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "app" {
  family                   = "stock-predictor"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([
    {
      name  = "stock-predictor"
      image = "your-account.dkr.ecr.us-west-2.amazonaws.com/stock-predictor:latest"
      
      portMappings = [
        {
          containerPort = 8501
          hostPort      = 8501
        }
      ]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.app.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

# ECS Service
resource "aws_ecs_service" "main" {
  name            = "stock-predictor-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.public[*].id
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "stock-predictor"
    container_port   = 8501
  }

  depends_on = [aws_lb_listener.app]
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "stock-predictor-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = {
    Environment = var.environment
  }
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier     = "stock-predictor-db"
  engine         = "postgres"
  engine_version = "13.7"
  instance_class = "db.t3.micro"
  allocated_storage = 20

  db_name  = "stockdb"
  username = "dbuser"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true

  tags = {
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "main" {
  cluster_id           = "stock-predictor-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis6.x"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]

  tags = {
    Environment = var.environment
  }
}
"""

# GitHub Actions CI/CD Pipeline
GITHUB_ACTIONS = """
name: Deploy Stock Predictor

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  AWS_REGION: us-west-2
  ECR_REPOSITORY: stock-predictor

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src/ --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}

    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1

    - name: Build, tag, and push image to Amazon ECR
      id: build-image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        echo "::set-output name=image::$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"

    - name: Deploy to ECS
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        aws ecs update-service --cluster stock-predictor-cluster --service stock-predictor-service --force-new-deployment

  deploy-infrastructure:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
    
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v1
      with:
        terraform_version: 1.0.0
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Terraform Init
      run: terraform init
    
    - name: Terraform Plan
      run: terraform plan
    
    - name: Terraform Apply
      if: github.ref == 'refs/heads/main'
      run: terraform apply -auto-approve
"""

# FastAPI Production Service
FASTAPI_SERVICE = """
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import redis
import json
import logging

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction API",
    description="Production ML API for stock market predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Load models at startup
models = {}
scaler = None

@app.on_event("startup")
async def startup_event():
    global models, scaler
    try:
        models['rf'] = joblib.load('models/random_forest.pkl')
        models['xgb'] = joblib.load('models/xgboost.pkl')
        scaler = joblib.load('models/scaler.pkl')
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading models: {e}")

# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    features: dict

class PredictionResponse(BaseModel):
    symbol: str
    predicted_return: float
    confidence: float
    timestamp: datetime
    model_used: str

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Validate input
        if not models or not scaler:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Prepare features
        feature_names = [
            'current_price', 'price_change', 'sma_5', 'sma_20', 
            'rsi', 'volume_ratio', 'volatility'
        ]
        
        feature_vector = np.array([
            request.features.get(name, 0) for name in feature_names
        ]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Make predictions with both models
        rf_pred = models['rf'].predict(feature_vector_scaled)[0]
        xgb_pred = models['xgb'].predict(feature_vector_scaled)[0]
        
        # Ensemble prediction
        ensemble_pred = 0.4 * rf_pred + 0.6 * xgb_pred
        
        # Calculate confidence based on agreement
        confidence = 1.0 - abs(rf_pred - xgb_pred) / (abs(rf_pred) + abs(xgb_pred) + 1e-8)
        confidence = max(0.5, min(0.95, confidence))
        
        # Cache prediction
        cache_key = f"prediction:{request.symbol}:{int(datetime.now().timestamp())}"
        cache_data = {
            'symbol': request.symbol,
            'prediction': ensemble_pred,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        redis_client.setex(cache_key, 300, json.dumps(cache_data))
        
        return PredictionResponse(
            symbol=request.symbol,
            predicted_return=ensemble_pred,
            confidence=confidence,
            timestamp=datetime.now(),
            model_used="ensemble"
        )
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    results = []
    
    for pred_request in request.requests:
        try:
            result = await predict(pred_request)
            results.append(result)
        except Exception as e:
            results.append({
                "symbol": pred_request.symbol,
                "error": str(e)
            })
    
    return {"predictions": results}

# Get cached predictions
@app.get("/predictions/{symbol}")
async def get_predictions(symbol: str, limit: int = 10):
    try:
        # Get recent predictions from cache
        keys = redis_client.keys(f"prediction:{symbol}:*")
        predictions = []
        
        for key in sorted(keys, reverse=True)[:limit]:
            data = redis_client.get(key)
            if data:
                predictions.append(json.loads(data))
        
        return {"symbol": symbol, "predictions": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model performance metrics
@app.get("/metrics")
async def get_metrics():
    try:
        # Get model metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return {
            "model_performance": metadata.get('test_metrics', {}),
            "last_training": metadata.get('training_date'),
            "feature_importance": metadata.get('feature_importance', {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task for model retraining
@app.post("/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    def retrain():
        # Implement model retraining logic
        logging.info("Starting model retraining...")
        # Add your retraining code here
        
    background_tasks.add_task(retrain)
    return {"message": "Model retraining started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

def create_deployment_files():
    """Create all deployment configuration files."""
    files_to_create = {
        'Dockerfile': DOCKERFILE_CONTENT,
        'docker-compose.yml': DOCKER_COMPOSE_CONTENT,
        'k8s-deployment.yaml': KUBERNETES_DEPLOYMENT,
        'terraform/main.tf': TERRAFORM_AWS,
        '.github/workflows/deploy.yml': GITHUB_ACTIONS,
        'src/api/main.py': FASTAPI_SERVICE
    }
    
    import os
    
    for file_path, content in files_to_create.items():
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write file
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Created {file_path}")

if __name__ == "__main__":
    create_deployment_files()
    print("All deployment files created successfully!")
