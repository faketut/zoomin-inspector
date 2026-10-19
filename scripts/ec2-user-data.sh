#!/bin/bash
# EC2 user-data for the ZoomIn Inspector Graviton instance (COOL AMI, Ubuntu 24.04).
# Paste into the "User data" field at launch. Runs once as root.
set -eux
apt-get update
apt-get install -y docker.io docker-compose-plugin awscli git python3-venv
usermod -aG docker ubuntu || true
mkdir -p /app/data /opt/venvs
systemctl enable --now docker
echo "bootstrap done: $(date -u)"
