#!/usr/bin/env bash
# Pull demo media from S3 into the running container on the EC2 instance via SSM.
set -euo pipefail
REGION="${1:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
B="s3://smart-dairy-opencv26-${ACCOUNT}-${REGION}/assets"
IID=$(aws cloudformation describe-stack-resource --stack-name smart-dairy --logical-resource-id SmartDairyInstance --region "$REGION" --query StackResourceDetail.PhysicalResourceId --output text)
CMD="mkdir -p /opt/app/output /opt/app/video && aws s3 sync $B/ /opt/app/ --region $REGION && docker cp /opt/app/output/. smart-dairy:/app/output/ && docker cp /opt/app/video/. smart-dairy:/app/video/ 2>/dev/null; docker exec smart-dairy ls -la /app/output /app/video"
CID=$(aws ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript --parameters "commands=[\"$CMD\"]" --region "$REGION" --query Command.CommandId --output text)
echo "command $CID on $IID"
for i in $(seq 1 30); do
  S=$(aws ssm get-command-invocation --command-id "$CID" --instance-id "$IID" --region "$REGION" --query Status --output text 2>/dev/null || echo Pending)
  case "$S" in Success|Failed|Cancelled|TimedOut) break;; esac
  sleep 5
done
echo "status: $S"
aws ssm get-command-invocation --command-id "$CID" --instance-id "$IID" --region "$REGION" --query "[StandardOutputContent,StandardErrorContent]" --output text

