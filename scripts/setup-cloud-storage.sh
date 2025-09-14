#!/bin/bash
# STS Neural Agent - Cloud Storage Setup Script
# Automatically sets up persistent storage for different cloud providers

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Default configuration
DEFAULT_SIZE="100"
DEFAULT_PATH="/mnt/sts-data"

echo -e "${BLUE}☁️  STS Neural Agent - Cloud Storage Setup${NC}"
echo "==========================================="

# Parse arguments
CLOUD_PROVIDER=""
STORAGE_SIZE="$DEFAULT_SIZE"
MOUNT_PATH="$DEFAULT_PATH"

while [[ $# -gt 0 ]]; do
    case $1 in
        --aws)
            CLOUD_PROVIDER="aws"
            shift
            ;;
        --gcp)
            CLOUD_PROVIDER="gcp"
            shift
            ;;
        --azure)
            CLOUD_PROVIDER="azure"
            shift
            ;;
        --generic)
            CLOUD_PROVIDER="generic"
            shift
            ;;
        --size)
            STORAGE_SIZE="$2"
            shift 2
            ;;
        --path)
            MOUNT_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--aws|--gcp|--azure|--generic] [--size GB] [--path /mount/path]"
            echo ""
            echo "Options:"
            echo "  --aws      Setup AWS EBS volume"
            echo "  --gcp      Setup GCP Persistent Disk"
            echo "  --azure    Setup Azure Managed Disk"
            echo "  --generic  Setup generic directory structure"
            echo "  --size GB  Storage size in GB (default: 100)"
            echo "  --path     Mount path (default: /mnt/sts-data)"
            echo "  --help     Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-detect cloud provider if not specified
if [ -z "$CLOUD_PROVIDER" ]; then
    if curl -s -m 5 http://169.254.169.254/latest/meta-data/ >/dev/null 2>&1; then
        CLOUD_PROVIDER="aws"
        echo -e "${YELLOW}🔍 Auto-detected: AWS EC2${NC}"
    elif curl -s -m 5 -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/ >/dev/null 2>&1; then
        CLOUD_PROVIDER="gcp"
        echo -e "${YELLOW}🔍 Auto-detected: Google Cloud${NC}"
    elif curl -s -m 5 -H "Metadata: true" http://169.254.169.254/metadata/instance/ >/dev/null 2>&1; then
        CLOUD_PROVIDER="azure"
        echo -e "${YELLOW}🔍 Auto-detected: Azure${NC}"
    else
        CLOUD_PROVIDER="generic"
        echo -e "${YELLOW}🔍 No cloud provider detected, using generic setup${NC}"
    fi
fi

echo "📋 Configuration:"
echo "   Provider: $CLOUD_PROVIDER"
echo "   Size: ${STORAGE_SIZE}GB"
echo "   Path: $MOUNT_PATH"
echo ""

# Cloud-specific setup functions
setup_aws() {
    echo -e "${BLUE}🔧 Setting up AWS EBS storage...${NC}"
    
    # Get instance metadata
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
    REGION=${AZ%?}
    
    echo "Instance: $INSTANCE_ID"
    echo "Zone: $AZ"
    
    # Check if volume already exists
    VOLUME_ID=$(aws ec2 describe-volumes \
        --region "$REGION" \
        --filters "Name=tag:Name,Values=sts-training-data" "Name=availability-zone,Values=$AZ" \
        --query 'Volumes[0].VolumeId' --output text 2>/dev/null)
    
    if [ "$VOLUME_ID" = "None" ] || [ -z "$VOLUME_ID" ]; then
        echo -e "${YELLOW}📀 Creating new EBS volume (${STORAGE_SIZE}GB)...${NC}"
        VOLUME_ID=$(aws ec2 create-volume \
            --region "$REGION" \
            --size "$STORAGE_SIZE" \
            --volume-type gp3 \
            --availability-zone "$AZ" \
            --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=sts-training-data}]" \
            --query 'VolumeId' --output text)
        
        echo "Waiting for volume to be available..."
        aws ec2 wait volume-available --region "$REGION" --volume-ids "$VOLUME_ID"
    fi
    
    echo -e "${YELLOW}🔗 Attaching volume $VOLUME_ID...${NC}"
    aws ec2 attach-volume \
        --region "$REGION" \
        --volume-id "$VOLUME_ID" \
        --instance-id "$INSTANCE_ID" \
        --device /dev/xvdf 2>/dev/null || echo "Volume may already be attached"
    
    # Wait for device to be available
    echo "Waiting for device to be available..."
    while [ ! -e /dev/xvdf ]; do sleep 1; done
    
    setup_filesystem "/dev/xvdf"
}

setup_gcp() {
    echo -e "${BLUE}🔧 Setting up GCP Persistent Disk...${NC}"
    
    # Get instance metadata
    INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
    ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d'/' -f4)
    PROJECT=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)
    
    echo "Instance: $INSTANCE_NAME"
    echo "Zone: $ZONE"
    echo "Project: $PROJECT"
    
    # Check if disk already exists
    if ! gcloud compute disks describe sts-training-data --zone="$ZONE" >/dev/null 2>&1; then
        echo -e "${YELLOW}💽 Creating persistent disk (${STORAGE_SIZE}GB)...${NC}"
        gcloud compute disks create sts-training-data \
            --size="${STORAGE_SIZE}GB" \
            --zone="$ZONE" \
            --type=pd-ssd
    fi
    
    echo -e "${YELLOW}🔗 Attaching disk...${NC}"
    gcloud compute instances attach-disk "$INSTANCE_NAME" \
        --disk=sts-training-data \
        --zone="$ZONE" 2>/dev/null || echo "Disk may already be attached"
    
    # Wait for device
    echo "Waiting for device to be available..."
    while [ ! -e /dev/sdb ]; do sleep 1; done
    
    setup_filesystem "/dev/sdb"
}

setup_azure() {
    echo -e "${BLUE}🔧 Setting up Azure Managed Disk...${NC}"
    
    # Get instance metadata
    RESOURCE_GROUP=$(curl -s -H "Metadata: true" "http://169.254.169.254/metadata/instance/compute/resourceGroupName?api-version=2021-02-01&format=text")
    VM_NAME=$(curl -s -H "Metadata: true" "http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text")
    LOCATION=$(curl -s -H "Metadata: true" "http://169.254.169.254/metadata/instance/compute/location?api-version=2021-02-01&format=text")
    
    echo "VM: $VM_NAME"
    echo "Resource Group: $RESOURCE_GROUP"
    echo "Location: $LOCATION"
    
    # Check if disk exists
    if ! az disk show --resource-group "$RESOURCE_GROUP" --name sts-training-disk >/dev/null 2>&1; then
        echo -e "${YELLOW}💽 Creating managed disk (${STORAGE_SIZE}GB)...${NC}"
        az disk create \
            --resource-group "$RESOURCE_GROUP" \
            --name sts-training-disk \
            --size-gb "$STORAGE_SIZE" \
            --sku Premium_LRS \
            --location "$LOCATION"
    fi
    
    echo -e "${YELLOW}🔗 Attaching disk...${NC}"
    az vm disk attach \
        --resource-group "$RESOURCE_GROUP" \
        --vm-name "$VM_NAME" \
        --name sts-training-disk 2>/dev/null || echo "Disk may already be attached"
    
    # Wait for device (Azure typically uses /dev/sdc)
    echo "Waiting for device to be available..."
    while [ ! -e /dev/sdc ]; do sleep 1; done
    
    setup_filesystem "/dev/sdc"
}

setup_generic() {
    echo -e "${BLUE}🔧 Setting up generic directory structure...${NC}"
    
    # Create directory structure
    sudo mkdir -p "$MOUNT_PATH"
    sudo chown -R "$USER:$USER" "$MOUNT_PATH"
    
    setup_directories
}

setup_filesystem() {
    local DEVICE="$1"
    
    echo -e "${YELLOW}💾 Setting up filesystem on $DEVICE...${NC}"
    
    # Check if already formatted
    if ! sudo file -s "$DEVICE" | grep -q filesystem; then
        echo "Formatting $DEVICE..."
        sudo mkfs.ext4 -F "$DEVICE"
    fi
    
    # Create mount point
    sudo mkdir -p "$MOUNT_PATH"
    
    # Mount the device
    sudo mount "$DEVICE" "$MOUNT_PATH"
    sudo chown -R "$USER:$USER" "$MOUNT_PATH"
    
    # Add to fstab for persistent mounting
    DEVICE_UUID=$(sudo blkid -s UUID -o value "$DEVICE")
    if ! grep -q "$DEVICE_UUID" /etc/fstab; then
        echo "UUID=$DEVICE_UUID $MOUNT_PATH ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
    fi
    
    setup_directories
}

setup_directories() {
    echo -e "${YELLOW}📁 Creating directory structure...${NC}"
    
    mkdir -p "$MOUNT_PATH"/{models,logs,wandb,configs,backups}
    
    # Set permissions
    chmod 755 "$MOUNT_PATH"
    chmod -R 755 "$MOUNT_PATH"/{models,logs,wandb,configs,backups}
    
    echo -e "${GREEN}✅ Directory structure created:${NC}"
    ls -la "$MOUNT_PATH"
}

# Run setup based on provider
case $CLOUD_PROVIDER in
    aws)
        setup_aws
        ;;
    gcp)
        setup_gcp
        ;;
    azure)
        setup_azure
        ;;
    generic)
        setup_generic
        ;;
    *)
        echo -e "${RED}❌ Unknown cloud provider: $CLOUD_PROVIDER${NC}"
        exit 1
        ;;
esac

# Update .env file
echo -e "${YELLOW}⚙️  Updating .env configuration...${NC}"
if [ -f .env ]; then
    # Update existing .env
    sed -i.bak "s|^DATA_PATH=.*|DATA_PATH=$MOUNT_PATH|" .env
else
    # Create new .env from template
    cp .env.example .env
    sed -i.bak "s|^DATA_PATH=.*|DATA_PATH=$MOUNT_PATH|" .env
fi

# Verify setup
echo ""
echo -e "${BLUE}🔍 Verifying setup...${NC}"
df -h "$MOUNT_PATH"
echo ""
echo -e "${GREEN}✅ Cloud storage setup completed!${NC}"
echo ""
echo "📋 Next steps:"
echo "1. Configure .env file with your settings"
echo "2. Set WANDB_API_KEY in .env"
echo "3. Run: ./scripts/cloud-train.sh"
echo ""
echo "📂 Data will be stored in: $MOUNT_PATH"