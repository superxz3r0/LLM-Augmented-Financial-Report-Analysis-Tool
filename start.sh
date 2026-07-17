#!/usr/bin/env bash
set -euo pipefail

export FINSIGHT_S3_BUCKET=finrag-original-fillings

# secret come from SSM
export OPENAI_API_KEY=$(aws ssm get-parameter --name /finsight/OPENAI_API_KEY \
    --with-decryption --query Parameter.Value --output text)
export GEMINI_API_KEY=$(aws ssm get-parameter --name /finsight/GEMINI_API_KEY \
    --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")

# sync from S3 bucket
aws s3 sync "s3://$FINSIGHT_S3_BUCKET/filings" /mnt/data/filings
aws s3 sync "s3://$FINSIGHT_S3_BUCKET/sample"  /mnt/data/sample || true

docker compose up -d --build
echo "started. logs:  docker compose logs -f"