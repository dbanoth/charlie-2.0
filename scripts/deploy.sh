#!/bin/bash
# Manual deployment script for Livestock Advisor Chatbot

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-animated-flare-421518}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
SERVICE_NAME="livestock-advisor"
ARTIFACT_REGISTRY="$REGION-docker.pkg.dev/$PROJECT_ID/chatbot"

echo "Deploying Livestock Advisor to Cloud Run..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Get the current timestamp for tagging
TAG=$(date +%Y%m%d-%H%M%S)

# Configure Docker for Artifact Registry
echo "Configuring Docker authentication..."
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet

# Build the Docker image
echo "Building Docker image..."
docker build -t $ARTIFACT_REGISTRY/$SERVICE_NAME:$TAG -t $ARTIFACT_REGISTRY/$SERVICE_NAME:latest .

# Push to Artifact Registry
echo "Pushing to Artifact Registry..."
docker push $ARTIFACT_REGISTRY/$SERVICE_NAME:$TAG
docker push $ARTIFACT_REGISTRY/$SERVICE_NAME:latest

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image=$ARTIFACT_REGISTRY/$SERVICE_NAME:$TAG \
    --region=$REGION \
    --platform=managed \
    --allow-unauthenticated \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=10 \
    --timeout=300 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION" \
    --set-secrets="DB_HOST=db-host:latest,DB_PORT=db-port:latest,DB_USER=db-user:latest,DB_PASSWORD=db-password:latest,DB_NAME=db-name:latest" \
    --service-account="chatbot-sa@$PROJECT_ID.iam.gserviceaccount.com"

# Get the service URL
echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test with:"
echo "  curl $SERVICE_URL/health"
