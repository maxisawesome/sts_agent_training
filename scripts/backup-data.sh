#!/bin/bash
# STS Neural Agent - Data Backup Script
# Syncs training data to cloud storage

set -e

# Load environment variables
if [ -f .env ]; then
    source .env
fi

DATA_DIR="${DATA_PATH:-/mnt/sts-data}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}📦 STS Data Backup - $TIMESTAMP${NC}"
echo "======================================"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Data directory not found: $DATA_DIR${NC}"
    exit 1
fi

# Create local backup first
LOCAL_BACKUP_DIR="$DATA_DIR/backups/$TIMESTAMP"
mkdir -p "$LOCAL_BACKUP_DIR"

echo -e "${YELLOW}📂 Creating local backup...${NC}"
rsync -av --exclude='wandb' "$DATA_DIR/models/" "$LOCAL_BACKUP_DIR/models/" 2>/dev/null || true
rsync -av "$DATA_DIR/configs/" "$LOCAL_BACKUP_DIR/configs/" 2>/dev/null || true
cp -r "$DATA_DIR/logs"/*.log "$LOCAL_BACKUP_DIR/" 2>/dev/null || true

echo -e "${GREEN}✅ Local backup created: $LOCAL_BACKUP_DIR${NC}"

# Cloud-specific backup functions
backup_to_aws() {
    if [ -n "$S3_BACKUP_BUCKET" ] && command -v aws >/dev/null 2>&1; then
        echo -e "${YELLOW}☁️  Syncing to AWS S3...${NC}"
        aws s3 sync "$DATA_DIR/models" "s3://$S3_BACKUP_BUCKET/models/" --delete
        aws s3 sync "$DATA_DIR/configs" "s3://$S3_BACKUP_BUCKET/configs/"
        aws s3 cp "$DATA_DIR/logs" "s3://$S3_BACKUP_BUCKET/logs/$TIMESTAMP/" --recursive --exclude="*" --include="*.log"
        echo -e "${GREEN}✅ AWS S3 backup completed${NC}"
    fi
}

backup_to_gcp() {
    if [ -n "$GCS_BACKUP_BUCKET" ] && command -v gsutil >/dev/null 2>&1; then
        echo -e "${YELLOW}☁️  Syncing to Google Cloud Storage...${NC}"
        gsutil -m rsync -r -d "$DATA_DIR/models" "gs://$GCS_BACKUP_BUCKET/models/"
        gsutil -m rsync -r "$DATA_DIR/configs" "gs://$GCS_BACKUP_BUCKET/configs/"
        gsutil -m cp "$DATA_DIR/logs"/*.log "gs://$GCS_BACKUP_BUCKET/logs/$TIMESTAMP/" 2>/dev/null || true
        echo -e "${GREEN}✅ Google Cloud Storage backup completed${NC}"
    fi
}

backup_to_azure() {
    if [ -n "$AZURE_STORAGE_ACCOUNT" ] && [ -n "$AZURE_CONTAINER" ] && command -v az >/dev/null 2>&1; then
        echo -e "${YELLOW}☁️  Syncing to Azure Blob Storage...${NC}"
        az storage blob upload-batch \
            --destination "$AZURE_CONTAINER/models" \
            --source "$DATA_DIR/models" \
            --account-name "$AZURE_STORAGE_ACCOUNT" 2>/dev/null || true
        az storage blob upload-batch \
            --destination "$AZURE_CONTAINER/configs" \
            --source "$DATA_DIR/configs" \
            --account-name "$AZURE_STORAGE_ACCOUNT" 2>/dev/null || true
        echo -e "${GREEN}✅ Azure Blob Storage backup completed${NC}"
    fi
}

# Run cloud backups
backup_to_aws
backup_to_gcp  
backup_to_azure

# Clean up old local backups (keep last 5)
echo -e "${YELLOW}🧹 Cleaning up old backups...${NC}"
cd "$DATA_DIR/backups" && ls -t | tail -n +6 | xargs rm -rf 2>/dev/null || true

# Backup summary
echo ""
echo -e "${BLUE}📊 Backup Summary${NC}"
echo "=================="
echo "📂 Models: $(ls -1 "$DATA_DIR/models" 2>/dev/null | wc -l) files ($(du -sh "$DATA_DIR/models" 2>/dev/null | cut -f1 || echo "0"))"
echo "📋 Configs: $(ls -1 "$DATA_DIR/configs" 2>/dev/null | wc -l) files"
echo "📝 Logs: $(ls -1 "$DATA_DIR/logs"/*.log 2>/dev/null | wc -l || echo "0") files"
echo "💾 Local backups: $(ls -1 "$DATA_DIR/backups" 2>/dev/null | wc -l) versions"
echo ""
echo -e "${GREEN}✅ Backup completed successfully!${NC}"