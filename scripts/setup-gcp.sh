#!/bin/bash
# GCP Setup Script for Livestock Advisor Chatbot
# Run this once to set up all required GCP resources

set -e

# Configuration - UPDATE THESE
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
SERVICE_ACCOUNT_NAME="chatbot-sa"
ARTIFACT_REGISTRY_REPO="chatbot"

echo "Setting up GCP resources for Livestock Advisor..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    firestore.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com

# Create Artifact Registry repository
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create $ARTIFACT_REGISTRY_REPO \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker images for chatbot" \
    2>/dev/null || echo "Repository already exists"

# Create Firestore database (Native mode)
echo "Creating Firestore database..."
gcloud firestore databases create \
    --location=$REGION \
    2>/dev/null || echo "Firestore database already exists"

# Create Firestore vector index
echo "Creating Firestore vector index..."
gcloud firestore indexes composite create \
    --collection-group=livestock_knowledge \
    --field-config=vector-config='{"dimension":"768","flat":"{}"}',field-path=embedding \
    2>/dev/null || echo "Index already exists or being created"

# Create service account
echo "Creating service account..."
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --display-name="Chatbot Service Account" \
    2>/dev/null || echo "Service account already exists"

SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

# Grant required roles
echo "Granting IAM roles..."
for ROLE in \
    "roles/aiplatform.user" \
    "roles/datastore.user" \
    "roles/secretmanager.secretAccessor" \
    "roles/logging.logWriter"
do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
        --role="$ROLE" \
        --quiet
done

# Create secrets for database credentials
echo "Creating secrets..."
echo -n "Enter DB_HOST: " && read DB_HOST
echo -n "Enter DB_PORT [1433]: " && read DB_PORT
DB_PORT=${DB_PORT:-1433}
echo -n "Enter DB_USER: " && read DB_USER
echo -n "Enter DB_PASSWORD: " && read -s DB_PASSWORD
echo ""
echo -n "Enter DB_NAME: " && read DB_NAME

# Create secrets
echo -n "$DB_HOST" | gcloud secrets create db-host --data-file=- 2>/dev/null || \
    echo -n "$DB_HOST" | gcloud secrets versions add db-host --data-file=-

echo -n "$DB_PORT" | gcloud secrets create db-port --data-file=- 2>/dev/null || \
    echo -n "$DB_PORT" | gcloud secrets versions add db-port --data-file=-

echo -n "$DB_USER" | gcloud secrets create db-user --data-file=- 2>/dev/null || \
    echo -n "$DB_USER" | gcloud secrets versions add db-user --data-file=-

echo -n "$DB_PASSWORD" | gcloud secrets create db-password --data-file=- 2>/dev/null || \
    echo -n "$DB_PASSWORD" | gcloud secrets versions add db-password --data-file=-

echo -n "$DB_NAME" | gcloud secrets create db-name --data-file=- 2>/dev/null || \
    echo -n "$DB_NAME" | gcloud secrets versions add db-name --data-file=-

# Grant secret access to service account
for SECRET in db-host db-port db-user db-password db-name; do
    gcloud secrets add-iam-policy-binding $SECRET \
        --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet
done

# Create service account key for local development (optional)
echo "Creating service account key..."
mkdir -p credentials
gcloud iam service-accounts keys create credentials/service-account.json \
    --iam-account=$SERVICE_ACCOUNT_EMAIL

echo ""
echo "========================================"
echo "GCP Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Set up Cloud Build trigger:"
echo "   gcloud builds triggers create github \\"
echo "     --repo-name=YOUR_REPO --repo-owner=YOUR_OWNER \\"
echo "     --branch-pattern='^main$' \\"
echo "     --build-config=cloudbuild.yaml"
echo ""
echo "2. Or deploy manually:"
echo "   ./scripts/deploy.sh"
echo ""
echo "3. Service account key saved to: credentials/service-account.json"
echo "   Update your .env file: GOOGLE_APPLICATION_CREDENTIALS=./credentials/service-account.json"
