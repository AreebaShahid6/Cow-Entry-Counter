#!/usr/bin/env bash
# One-command deploy of Smart Dairy to AWS (EC2 Graviton + Docker).
# Requires: aws CLI configured (aws configure / aws sso login).
# Usage: ./deploy.sh [region]   (default us-east-1)
set -euo pipefail
cd "$(dirname "$0")"

REGION="${1:-us-east-1}"
STACK="smart-dairy"
export AWS_DEFAULT_REGION="$REGION"

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="smart-dairy-opencv26-${ACCOUNT}-${REGION}"

echo ">> Deploying stack $STACK in $REGION (account $ACCOUNT)"
aws cloudformation deploy --stack-name "$STACK" \
  --template-file aws_cloudformation.yaml \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --parameter-overrides GravitonInstanceType=m7i-flex.large

echo ">> Packaging and uploading code bundle"
tar czf /tmp/smart-dairy-app.tgz \
  --exclude=venv --exclude=.git --exclude=__pycache__ \
  --exclude='output/devpost_thumbnail*' --exclude=tests .
aws s3 cp /tmp/smart-dairy-app.tgz "s3://${BUCKET}/deploy/app.tgz"

URL=$(aws cloudformation describe-stacks --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='WebsiteURL'].OutputValue" --output text)
echo ">> Done. The instance builds the image on boot (~5-10 min)."
echo ">> Website: $URL"

