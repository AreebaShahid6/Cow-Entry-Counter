#!/usr/bin/env bash
set -euo pipefail
IID=$(aws cloudformation describe-stack-resource --stack-name smart-dairy --logical-resource-id SmartDairyInstance --region us-east-1 --query StackResourceDetail.PhysicalResourceId --output text)
echo "$IID"
aws ec2 reboot-instances --instance-ids "$IID" --region us-east-1
echo rebooted

