#!/usr/bin/env bash
# Upload demo media (dashboard video, heatmap, CSVs) to the stack's S3 bucket.
set -euo pipefail
cd "$(dirname "$0")"
REGION="${1:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
B="s3://smart-dairy-opencv26-${ACCOUNT}-${REGION}/assets"
for f in output/cow_counter_dashboard.mp4 output/heatmap_final.png output/entries.csv output/cow_records.csv video/your_video.mp4; do
  aws s3 cp "$f" "$B/$f" --region "$REGION"
done
echo UPLOAD_DONE

