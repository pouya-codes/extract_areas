# AWS Cloud Deployment Roadmap

## Executive Summary

This document outlines the roadmap for deploying the AIMViewer + WSI Processing Platform to AWS, optimized for cloud-native architecture with shared storage access, containerized AI models, and horizontal scaling capabilities.

---

## Current Architecture vs. Target Architecture

### Current (On-Premise)
```
┌─────────────────────────────────────────────────────────────────┐
│                     Local Development Setup                      │
├─────────────────────────────────────────────────────────────────┤
│  • Django/OMERO Frontend (single server)                        │
│  • FastAPI Backend (single instance)                            │
│  • AI Models loaded in same process                             │
│  • Local filesystem for slide storage                           │
│  • Images passed as base64/bytes between components             │
└─────────────────────────────────────────────────────────────────┘
```

### Target (AWS Cloud-Native)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AWS Cloud Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │   CloudFront     │───▶│  ALB (Frontend)  │───▶│  ALB (API)       │       │
│  │   CDN            │    └────────┬─────────┘    └────────┬─────────┘       │
│  └──────────────────┘             │                       │                  │
│                                   ▼                       ▼                  │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                         EKS Cluster                               │       │
│  │  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────┐ │       │
│  │  │ AIMViewer      │  │ FastAPI        │  │ AI Model Workers    │ │       │
│  │  │ Frontend Pods  │  │ Backend Pods   │  │ (Auto-scaling)      │ │       │
│  │  │ (OMERO Web)    │  │ (API Gateway)  │  │ - DeepLIIF Pod      │ │       │
│  │  │ Replicas: 2-5  │  │ Replicas: 2-10 │  │ - HoVerNet Pod      │ │       │
│  │  └────────────────┘  └────────────────┘  │ - EC Cancer Pod     │ │       │
│  │                                          │ - SAM2 Pod          │ │       │
│  │                                          └─────────────────────┘ │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                     Shared Storage Layer                          │       │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │       │
│  │  │ Amazon EFS      │  │ Amazon S3       │  │ ElastiCache     │   │       │
│  │  │ (WSI Slides)    │  │ (Results/Cache) │  │ (Redis Queue)   │   │       │
│  │  │ Mount: /slides  │  │ Bucket: results │  │                 │   │       │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘   │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                   │                                          │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                     Data Services                                 │       │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │       │
│  │  │ RDS PostgreSQL  │  │ OMERO Server    │  │ Amazon SQS      │   │       │
│  │  │ (Metadata)      │  │ (Ice Protocol)  │  │ (Job Queue)     │   │       │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘   │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation & Shared Storage (Weeks 1-3)

### 1.1 Shared Storage Setup with Amazon EFS

**Objective:** Enable all services to access WSI slides via shared filesystem

**Tasks:**
- [ ] Create Amazon EFS filesystem with appropriate throughput mode
- [ ] Configure EFS mount targets in each Availability Zone
- [ ] Set up EFS access points for different components
- [ ] Migrate existing slides to EFS
- [ ] Configure security groups for EFS access

**EFS Configuration:**
```yaml
# terraform/efs.tf
resource "aws_efs_file_system" "wsi_slides" {
  creation_token = "wsi-slides-efs"
  performance_mode = "generalPurpose"
  throughput_mode = "bursting"  # or "provisioned" for high-throughput
  
  tags = {
    Name = "WSI-Slides-Storage"
  }
}

resource "aws_efs_mount_target" "slides_mount" {
  for_each        = toset(var.availability_zones)
  file_system_id  = aws_efs_file_system.wsi_slides.id
  subnet_id       = var.private_subnets[each.key]
  security_groups = [aws_security_group.efs_sg.id]
}
```

### 1.2 Refactor API to Use File Paths Instead of Image Transfer

**Current Flow:**
```
Frontend → Extract Image → Base64 Encode → Send to API → Decode → Process
```

**Optimized Cloud Flow:**
```
Frontend → Send {slide_path, region_bbox, annotation_points} → API reads directly from EFS
```

**API Changes Required:**

```python
# app_refactored_cloud.py - New cloud-optimized endpoints

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
import openslide  # For reading WSI directly

class RegionRequest(BaseModel):
    slide_path: str              # e.g., "/slides/OV_van/VOA-240-A.tiff"
    region_bbox: List[int]       # [x, y, width, height] in slide coordinates
    annotation_points: List[Tuple[float, float]]  # Polygon points
    model_name: str
    zoom_level: int = 0          # 0 = highest resolution
    hyperparameters: Optional[dict] = None

@app.post("/v2/process_region")
async def process_region_cloud(request: RegionRequest):
    """
    Cloud-optimized endpoint: reads slide directly from shared storage.
    No image transfer overhead - only metadata is passed.
    """
    # Validate slide exists on EFS
    slide_full_path = f"/mnt/efs/slides{request.slide_path}"
    if not Path(slide_full_path).exists():
        raise HTTPException(404, f"Slide not found: {request.slide_path}")
    
    # Open slide directly
    slide = openslide.OpenSlide(slide_full_path)
    
    # Extract region at specified zoom level
    x, y, w, h = request.region_bbox
    region_image = slide.read_region((x, y), request.zoom_level, (w, h))
    region_image = region_image.convert("RGB")
    
    # Create mask from annotation points
    mask = create_mask_from_points(request.annotation_points, (w, h))
    
    # Process with AI model
    model = model_registry.get_model(request.model_name)
    result = model.process(region_image, mask)
    
    # Store result in S3 instead of returning inline
    result_key = save_result_to_s3(result, request)
    
    return {
        "success": True,
        "result_url": f"https://s3.amazonaws.com/results/{result_key}",
        "scores": result.get("scores", {}),
        "str_result": result.get("str_result", "")
    }
```

### 1.3 S3 Setup for Results & Caching

**Tasks:**
- [ ] Create S3 bucket for processed results
- [ ] Configure S3 lifecycle policies (expire after 30 days)
- [ ] Set up CloudFront distribution for fast result delivery
- [ ] Implement result caching to avoid reprocessing

```yaml
# terraform/s3.tf
resource "aws_s3_bucket" "results" {
  bucket = "aimviewer-results-${var.environment}"
}

resource "aws_s3_bucket_lifecycle_configuration" "results_lifecycle" {
  bucket = aws_s3_bucket.results.id
  
  rule {
    id     = "expire-old-results"
    status = "Enabled"
    
    expiration {
      days = 30
    }
  }
}
```

---

## Phase 2: Containerize AI Models (Weeks 4-6)

### 2.1 Docker Strategy - One Container Per Model

Each AI model will be containerized independently for:
- Independent scaling based on demand
- Isolated dependencies (different CUDA versions, libraries)
- Easier updates and rollbacks
- GPU resource optimization

**Directory Structure:**
```
docker/
├── base/
│   └── Dockerfile.gpu-base       # Shared CUDA/Python base
├── deepliif/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── entrypoint.sh
├── hovernet/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── entrypoint.sh
├── ec-cancer/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── entrypoint.sh
├── sam2/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── entrypoint.sh
├── api-gateway/
│   ├── Dockerfile
│   └── requirements.txt
└── docker-compose.yml
```

### 2.2 Base GPU Image

```dockerfile
# docker/base/Dockerfile.gpu-base
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and common dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenslide0 \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

# Common Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    pillow \
    opencv-python-headless \
    openslide-python \
    fastapi \
    uvicorn \
    httpx \
    pydantic

WORKDIR /app
```

### 2.3 DeepLIIF Model Container

```dockerfile
# docker/deepliif/Dockerfile
FROM aimviewer-gpu-base:latest

# DeepLIIF specific dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy model code
COPY models/deepliif_model.py /app/models/
COPY models/base_model.py /app/models/
COPY module/DeepLiff/ /app/module/DeepLiff/

# Model weights will be mounted via volume
ENV MODEL_WEIGHTS_PATH=/models/deepliif

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8001/health || exit 1

# Run as model microservice
COPY docker/deepliif/model_service.py /app/
EXPOSE 8001
CMD ["uvicorn", "model_service:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 2.4 Model Microservice Template

Each model runs as an independent microservice:

```python
# docker/deepliif/model_service.py
"""
DeepLIIF Model Microservice

Runs as a standalone FastAPI service that:
1. Receives image paths (not images) from the API gateway
2. Reads images directly from EFS
3. Processes with DeepLIIF
4. Returns results to S3
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
from pathlib import Path
import openslide
from PIL import Image
import boto3
import os

from models.deepliif_model import DeepLIIFModel
from models.model_registry import model_registry

app = FastAPI(title="DeepLIIF Model Service")

# Initialize model on startup
@app.on_event("startup")
async def startup():
    model_registry.register_model_class("deepliif", DeepLIIFModel)
    model_registry.load_model("deepliif", {
        'model_dir': os.environ.get('MODEL_WEIGHTS_PATH', '/models/deepliif'),
        'tile_size': 256,
        'post_processing': True,
        'gpu_ids': [0] if os.environ.get('NVIDIA_VISIBLE_DEVICES') else []
    })
    print("✓ DeepLIIF model loaded")


class ProcessRequest(BaseModel):
    slide_path: str
    region_bbox: List[int]
    annotation_points: List[Tuple[float, float]]
    result_s3_key: str
    zoom_level: int = 0
    hyperparameters: Optional[dict] = None


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "deepliif", "gpu": True}


@app.post("/process")
async def process(request: ProcessRequest):
    """Process a region with DeepLIIF model."""
    # Read from shared EFS mount
    slide_path = Path(f"/mnt/efs/slides{request.slide_path}")
    if not slide_path.exists():
        raise HTTPException(404, f"Slide not found: {request.slide_path}")
    
    # Extract region
    slide = openslide.OpenSlide(str(slide_path))
    x, y, w, h = request.region_bbox
    region = slide.read_region((x, y), request.zoom_level, (w, h)).convert("RGB")
    
    # Create mask
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.polygon(request.annotation_points, fill=255)
    
    # Process
    model = model_registry.get_model("deepliif")
    result = model.process(
        region, 
        mask, 
        hyperparameters=request.hyperparameters
    )
    
    # Upload result to S3
    s3 = boto3.client('s3')
    # ... upload processed_image and scores
    
    return {
        "success": True,
        "scores": result.get("scores", {}),
        "str_result": result.get("str_result", ""),
        "result_url": f"s3://aimviewer-results/{request.result_s3_key}"
    }
```

### 2.5 Build & Push Pipeline (GitHub Actions)

```yaml
# .github/workflows/build-models.yml
name: Build AI Model Containers

on:
  push:
    paths:
      - 'models/**'
      - 'docker/**'
    branches: [main]

env:
  AWS_REGION: us-west-2
  ECR_REGISTRY: ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.us-west-2.amazonaws.com

jobs:
  build-base:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Build base image
        run: |
          docker build -t aimviewer-gpu-base:latest -f docker/base/Dockerfile.gpu-base .
          
  build-models:
    needs: build-base
    strategy:
      matrix:
        model: [deepliif, hovernet, ec-cancer, sam2]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push ${{ matrix.model }}
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker build -t $ECR_REGISTRY/aimviewer-${{ matrix.model }}:${{ github.sha }} \
            -f docker/${{ matrix.model }}/Dockerfile .
          docker push $ECR_REGISTRY/aimviewer-${{ matrix.model }}:${{ github.sha }}
```

---

## Phase 3: Kubernetes Deployment (Weeks 7-9)

### 3.1 EKS Cluster Setup

**Tasks:**
- [ ] Create EKS cluster with GPU node group
- [ ] Install NVIDIA device plugin
- [ ] Configure Cluster Autoscaler
- [ ] Set up EFS CSI driver for shared storage
- [ ] Configure AWS Load Balancer Controller

```yaml
# terraform/eks.tf
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "aimviewer-cluster"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    # CPU nodes for frontend/API
    general = {
      instance_types = ["m5.xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
    }
    
    # GPU nodes for AI models
    gpu = {
      ami_type       = "AL2_x86_64_GPU"
      instance_types = ["g4dn.xlarge"]  # 1 GPU, 4 vCPU, 16GB
      min_size       = 1
      max_size       = 8
      desired_size   = 2
      
      labels = {
        "nvidia.com/gpu" = "true"
        "workload"       = "ai-inference"
      }
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}
```

### 3.2 Kubernetes Manifests

**DeepLIIF Deployment with GPU & EFS:**

```yaml
# k8s/deployments/deepliif.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepliif-model
  namespace: aimviewer
spec:
  replicas: 2
  selector:
    matchLabels:
      app: deepliif
  template:
    metadata:
      labels:
        app: deepliif
    spec:
      nodeSelector:
        nvidia.com/gpu: "true"
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
      containers:
        - name: deepliif
          image: ${ECR_REGISTRY}/aimviewer-deepliif:latest
          ports:
            - containerPort: 8001
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "8Gi"
            requests:
              memory: "4Gi"
              cpu: "2"
          volumeMounts:
            - name: slides
              mountPath: /mnt/efs/slides
              readOnly: true
            - name: model-weights
              mountPath: /models
          env:
            - name: AWS_REGION
              value: "us-west-2"
            - name: S3_RESULTS_BUCKET
              value: "aimviewer-results"
          livenessProbe:
            httpGet:
              path: /health
              port: 8001
            initialDelaySeconds: 60
            periodSeconds: 30
      volumes:
        - name: slides
          persistentVolumeClaim:
            claimName: efs-slides-pvc
        - name: model-weights
          persistentVolumeClaim:
            claimName: model-weights-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: deepliif-service
  namespace: aimviewer
spec:
  selector:
    app: deepliif
  ports:
    - port: 8001
      targetPort: 8001
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: deepliif-hpa
  namespace: aimviewer
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deepliif-model
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: gpu_utilization
        target:
          type: AverageValue
          averageValue: "70"
```

### 3.3 API Gateway Deployment

```yaml
# k8s/deployments/api-gateway.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: aimviewer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
        - name: api
          image: ${ECR_REGISTRY}/aimviewer-api:latest
          ports:
            - containerPort: 8000
          env:
            - name: DEEPLIIF_SERVICE_URL
              value: "http://deepliif-service:8001"
            - name: HOVERNET_SERVICE_URL
              value: "http://hovernet-service:8002"
            - name: EC_CANCER_SERVICE_URL
              value: "http://ec-cancer-service:8003"
            - name: SAM2_SERVICE_URL
              value: "http://sam2-service:8004"
          volumeMounts:
            - name: slides
              mountPath: /mnt/efs/slides
              readOnly: true
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2"
      volumes:
        - name: slides
          persistentVolumeClaim:
            claimName: efs-slides-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
  namespace: aimviewer
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: LoadBalancer
  selector:
    app: api-gateway
  ports:
    - port: 80
      targetPort: 8000
```

---

## Phase 4: Job Queue System (Weeks 10-11)

### 4.1 Asynchronous Processing with SQS + Workers

For long-running AI processing, implement async job queue:

```
User Request → API → SQS Queue → Worker Pods → S3 Results → WebSocket Notification
```

```python
# api_gateway/job_manager.py
import boto3
import json
import uuid
from datetime import datetime

sqs = boto3.client('sqs')
QUEUE_URL = os.environ['SQS_QUEUE_URL']

class JobManager:
    async def submit_job(self, request: ProcessRequest) -> str:
        """Submit processing job to queue and return job ID."""
        job_id = str(uuid.uuid4())
        
        message = {
            'job_id': job_id,
            'model_name': request.model_name,
            'slide_path': request.slide_path,
            'region_bbox': request.region_bbox,
            'annotation_points': request.annotation_points,
            'submitted_at': datetime.utcnow().isoformat(),
            'result_s3_key': f"results/{job_id}"
        }
        
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(message),
            MessageGroupId=request.model_name  # FIFO queue grouping
        )
        
        return job_id
    
    async def get_job_status(self, job_id: str) -> dict:
        """Check job status from DynamoDB."""
        # Query DynamoDB for job status
        pass
```

### 4.2 Worker Service

```python
# workers/model_worker.py
"""
Model Worker - Polls SQS and processes jobs
Runs as a Kubernetes Deployment with GPU access
"""

import asyncio
import boto3
import json
import httpx
from typing import Dict

sqs = boto3.client('sqs')
dynamodb = boto3.resource('dynamodb')

MODEL_SERVICES = {
    'deepliif': 'http://deepliif-service:8001',
    'hovernet': 'http://hovernet-service:8002',
    'ec_cancer': 'http://ec-cancer-service:8003',
    'sam2': 'http://sam2-service:8004',
}


async def process_message(message: Dict):
    """Process a single job message."""
    body = json.loads(message['Body'])
    job_id = body['job_id']
    model_name = body['model_name']
    
    # Update status to "processing"
    update_job_status(job_id, 'processing')
    
    try:
        # Call appropriate model service
        service_url = MODEL_SERVICES[model_name]
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{service_url}/process",
                json={
                    'slide_path': body['slide_path'],
                    'region_bbox': body['region_bbox'],
                    'annotation_points': body['annotation_points'],
                    'result_s3_key': body['result_s3_key'],
                }
            )
            result = response.json()
        
        # Update status to "completed"
        update_job_status(job_id, 'completed', result)
        
    except Exception as e:
        update_job_status(job_id, 'failed', {'error': str(e)})


async def poll_queue():
    """Continuously poll SQS for messages."""
    while True:
        response = sqs.receive_message(
            QueueUrl=os.environ['SQS_QUEUE_URL'],
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=300  # 5 min timeout for processing
        )
        
        for message in response.get('Messages', []):
            await process_message(message)
            sqs.delete_message(
                QueueUrl=os.environ['SQS_QUEUE_URL'],
                ReceiptHandle=message['ReceiptHandle']
            )
```

---

## Phase 5: Frontend Integration (Weeks 12-13)

### 5.1 Update AIMViewer Frontend

Modify frontend to use cloud-optimized API:

```javascript
// frontend/annotator/src/api/cloudApi.js

const API_BASE = process.env.REACT_APP_API_URL || 'https://api.aimviewer.example.com';

/**
 * Cloud-optimized API client
 * Sends only paths and coordinates, not images
 */
export class CloudAPI {
  
  /**
   * Process a region with AI model
   * @param {Object} params
   * @param {string} params.slidePath - Path to slide on shared storage
   * @param {Array} params.regionBbox - [x, y, width, height]
   * @param {Array} params.annotationPoints - Polygon points
   * @param {string} params.modelName - AI model to use
   */
  async processRegion({ slidePath, regionBbox, annotationPoints, modelName }) {
    // For quick processing, use sync endpoint
    const response = await fetch(`${API_BASE}/v2/process_region`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        slide_path: slidePath,
        region_bbox: regionBbox,
        annotation_points: annotationPoints,
        model_name: modelName,
      })
    });
    
    return response.json();
  }
  
  /**
   * Submit async job for large regions
   * Returns job ID for status polling
   */
  async submitJob({ slidePath, regionBbox, annotationPoints, modelName }) {
    const response = await fetch(`${API_BASE}/v2/jobs/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        slide_path: slidePath,
        region_bbox: regionBbox,
        annotation_points: annotationPoints,
        model_name: modelName,
      })
    });
    
    return response.json();  // { job_id: "xxx", status: "queued" }
  }
  
  /**
   * Check job status
   */
  async getJobStatus(jobId) {
    const response = await fetch(`${API_BASE}/v2/jobs/${jobId}`);
    return response.json();
    // { status: "completed", result_url: "https://...", scores: {...} }
  }
  
  /**
   * WebSocket connection for real-time updates
   */
  connectWebSocket(onMessage) {
    const ws = new WebSocket(`wss://api.aimviewer.example.com/ws`);
    ws.onmessage = (event) => onMessage(JSON.parse(event.data));
    return ws;
  }
}
```

### 5.2 Update OMERO Integration

```python
# aimviewer/views_cloud.py
"""
Cloud-optimized views that work with shared storage
"""

from django.http import JsonResponse
from django.views import View
import httpx

class ProcessRegionView(View):
    """
    Cloud-optimized view: sends file path instead of image data
    """
    
    async def post(self, request, image_id):
        # Get slide path from OMERO
        slide_path = self.get_slide_path(image_id)  # Returns EFS path
        
        # Get annotation from request
        annotation = json.loads(request.POST.get('annotation'))
        
        # Call cloud API with path only
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.AI_API_URL}/v2/process_region",
                json={
                    'slide_path': slide_path,
                    'region_bbox': annotation['bbox'],
                    'annotation_points': annotation['points'],
                    'model_name': request.POST.get('model', 'deepliif')
                },
                timeout=300
            )
        
        return JsonResponse(response.json())
    
    def get_slide_path(self, image_id):
        """Get EFS path for an OMERO image."""
        # Query OMERO for original file path
        conn = get_omero_connection()
        image = conn.getObject("Image", image_id)
        
        # Original file path is accessible to all pods via EFS
        return image.getFileset().listFiles()[0].getPath()
```

---

## Phase 6: Monitoring & Operations (Weeks 14-15)

### 6.1 CloudWatch Metrics & Dashboards

```yaml
# terraform/monitoring.tf
resource "aws_cloudwatch_dashboard" "aimviewer" {
  dashboard_name = "AIMViewer-Operations"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "API Request Latency"
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", "${aws_lb.api.arn_suffix}"]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Model Processing Queue Depth"
          metrics = [
            ["AWS/SQS", "ApproximateNumberOfMessagesVisible", "QueueName", "aimviewer-jobs"]
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 24
        height = 6
        properties = {
          title   = "GPU Utilization by Model"
          metrics = [
            ["ContainerInsights", "pod_gpu_utilization", "PodName", "deepliif-*"],
            ["ContainerInsights", "pod_gpu_utilization", "PodName", "hovernet-*"],
            ["ContainerInsights", "pod_gpu_utilization", "PodName", "ec-cancer-*"]
          ]
        }
      }
    ]
  })
}
```

### 6.2 Alerting

```yaml
# terraform/alerts.tf
resource "aws_cloudwatch_metric_alarm" "high_queue_depth" {
  alarm_name          = "aimviewer-high-queue-depth"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = "300"
  statistic           = "Average"
  threshold           = "50"
  alarm_description   = "Job queue depth is high - may need more GPU workers"
  
  dimensions = {
    QueueName = "aimviewer-jobs"
  }
  
  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "model_errors" {
  alarm_name          = "aimviewer-model-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "5XXError"
  namespace           = "AWS/ApplicationELB"
  period              = "60"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "Model services returning errors"
  
  alarm_actions = [aws_sns_topic.alerts.arn]
}
```

---

## Cost Estimation

| Resource | Configuration | Monthly Cost (Est.) |
|----------|--------------|---------------------|
| EKS Cluster | 1 cluster | $73 |
| EC2 - General (m5.xlarge) | 3 instances avg | $420 |
| EC2 - GPU (g4dn.xlarge) | 2-4 instances avg | $630-1,260 |
| EFS | 500GB, bursting | $150 |
| S3 | 100GB results | $3 |
| RDS PostgreSQL | db.t3.medium | $50 |
| ALB | 2 load balancers | $40 |
| Data Transfer | 500GB/month | $45 |
| **Total (Low)** | | **~$1,400/month** |
| **Total (High)** | | **~$2,100/month** |

**Cost Optimization Tips:**
- Use Spot instances for GPU workers (up to 70% savings)
- Schedule scale-down during off-hours
- Use Savings Plans for baseline capacity

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | Weeks 1-3 | EFS setup, API refactor for path-based access, S3 results storage |
| Phase 2 | Weeks 4-6 | Docker containers for each model, ECR registry, CI/CD pipeline |
| Phase 3 | Weeks 7-9 | EKS cluster, GPU nodes, Kubernetes deployments, auto-scaling |
| Phase 4 | Weeks 10-11 | SQS job queue, async workers, DynamoDB job tracking |
| Phase 5 | Weeks 12-13 | Frontend updates, WebSocket notifications, testing |
| Phase 6 | Weeks 14-15 | Monitoring, alerting, documentation, production cutover |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU instance availability | Use multiple instance types (g4dn, g5), enable Spot with fallback to On-Demand |
| Model loading time | Use warm pools, pre-pull images to nodes, keep min 1 replica always running |
| Large slide file access | EFS throughput provisioning, consider FSx for Lustre for very high throughput |
| Cost overruns | Implement resource quotas, scheduled scaling, budget alerts |
| Data privacy (PHI) | Use private subnets, VPC endpoints, encrypt EFS/S3, enable audit logging |

---

## Next Steps

1. **Immediate (Week 1):**
   - [ ] Set up AWS account and VPC infrastructure
   - [ ] Create Terraform configuration for core resources
   - [ ] Begin refactoring API for path-based access

2. **Short-term (Weeks 2-4):**
   - [ ] Complete EFS setup and slide migration plan
   - [ ] Create first Docker container (DeepLIIF)
   - [ ] Test EFS mount in Docker locally

3. **Medium-term (Weeks 5-8):**
   - [ ] Complete all model containers
   - [ ] Set up EKS cluster
   - [ ] Deploy first model to Kubernetes

---

## Appendix: Quick Reference Commands

```bash
# Build all Docker images
docker-compose -f docker/docker-compose.yml build

# Push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-west-2.amazonaws.com
docker push <account>.dkr.ecr.us-west-2.amazonaws.com/aimviewer-deepliif:latest

# Deploy to EKS
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/storage/
kubectl apply -f k8s/deployments/

# Scale GPU workers
kubectl scale deployment deepliif-model --replicas=4 -n aimviewer

# View logs
kubectl logs -f deployment/deepliif-model -n aimviewer

# Check GPU utilization
kubectl exec -it <pod-name> -n aimviewer -- nvidia-smi
```
