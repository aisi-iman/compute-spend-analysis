#!/bin/bash
#
# EC2 Cost Optimization - Weekly Report Automation
#
# This script:
# 1. Runs the EC2 cost analyzer to generate fresh data
# 2. Executes the Jupyter notebook to generate the report
# 3. Exports the notebook to HTML for sharing
# 4. Optionally uploads to S3 or sends via email
#
# Usage:
#   ./run_weekly_report.sh
#
# For automation, add to crontab:
#   0 9 * * 1 /path/to/run_weekly_report.sh
#   (Runs every Monday at 9 AM)
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATE_STAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DATE=$(date +%Y-%m-%d)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================================================="
echo "EC2 Cost Optimization Report - Weekly Automation"
echo "=============================================================================="
echo "Date: $REPORT_DATE"
echo "Project: $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

# Step 1: Generate cost data
echo -e "${YELLOW}[1/4]${NC} Generating cost analysis data..."
if uv run ec2_cost_analyzer.py; then
    echo -e "${GREEN}✓${NC} Cost data generated successfully"

    # Find the most recent JSON report
    LATEST_REPORT=$(ls -t ec2_cost_report_*.json 2>/dev/null | head -1)
    if [ -z "$LATEST_REPORT" ]; then
        echo -e "${RED}✗${NC} No cost report found!"
        exit 1
    fi
    echo "  Using: $LATEST_REPORT"
else
    echo -e "${RED}✗${NC} Failed to generate cost data"
    exit 1
fi

# Step 2: Update notebook configuration
echo ""
echo -e "${YELLOW}[2/4]${NC} Updating notebook configuration..."
TEMP_NOTEBOOK="ec2_cost_optimization_report_temp.ipynb"
cp ec2_cost_optimization_report.ipynb "$TEMP_NOTEBOOK"

# Update the REPORT_JSON_PATH in the notebook
# Note: This is a simple sed replacement. For production, consider using papermill with parameters
sed -i "s|REPORT_JSON_PATH = \".*\"|REPORT_JSON_PATH = \"$LATEST_REPORT\"|g" "$TEMP_NOTEBOOK"
echo -e "${GREEN}✓${NC} Configuration updated"

# Step 3: Execute notebook
echo ""
echo -e "${YELLOW}[3/4]${NC} Executing Jupyter notebook..."
OUTPUT_NOTEBOOK="ec2_cost_report_executed_${DATE_STAMP}.ipynb"
if uv run jupyter nbconvert \
    --to notebook \
    --execute \
    --output "$OUTPUT_NOTEBOOK" \
    --ExecutePreprocessor.timeout=600 \
    "$TEMP_NOTEBOOK"; then
    echo -e "${GREEN}✓${NC} Notebook executed successfully"
    echo "  Output: $OUTPUT_NOTEBOOK"
else
    echo -e "${RED}✗${NC} Notebook execution failed"
    rm -f "$TEMP_NOTEBOOK"
    exit 1
fi

# Clean up temp notebook
rm -f "$TEMP_NOTEBOOK"

# Step 4: Export to HTML
echo ""
echo -e "${YELLOW}[4/4]${NC} Exporting to HTML..."
OUTPUT_HTML="ec2_cost_report_${DATE_STAMP}.html"
if uv run jupyter nbconvert \
    --to html \
    --output "$OUTPUT_HTML" \
    "$OUTPUT_NOTEBOOK"; then
    echo -e "${GREEN}✓${NC} HTML report generated"
    echo "  Output: $OUTPUT_HTML"
else
    echo -e "${RED}✗${NC} HTML export failed"
    exit 1
fi

# Optional: Upload to S3
# Uncomment and configure if you want to upload to S3
# echo ""
# echo "Uploading to S3..."
# S3_BUCKET="your-bucket-name"
# S3_PATH="cost-reports/$REPORT_DATE/"
# aws s3 cp "$OUTPUT_HTML" "s3://$S3_BUCKET/$S3_PATH$OUTPUT_HTML"
# aws s3 cp "$OUTPUT_NOTEBOOK" "s3://$S3_BUCKET/$S3_PATH$OUTPUT_NOTEBOOK"
# echo "✓ Uploaded to S3: s3://$S3_BUCKET/$S3_PATH"

# Optional: Send email notification
# Uncomment and configure if you want email notifications
# echo ""
# echo "Sending email notification..."
# EMAIL_TO="team@example.com"
# EMAIL_SUBJECT="EC2 Cost Optimization Report - $REPORT_DATE"
# EMAIL_BODY="Weekly EC2 cost optimization report is ready. View the attached HTML file."
#
# aws ses send-email \
#     --from "noreply@example.com" \
#     --to "$EMAIL_TO" \
#     --subject "$EMAIL_SUBJECT" \
#     --text "$EMAIL_BODY" \
#     --html "file://$OUTPUT_HTML"

# Summary
echo ""
echo "=============================================================================="
echo -e "${GREEN}Report Generation Complete!${NC}"
echo "=============================================================================="
echo ""
echo "Generated files:"
echo "  - Cost data: $LATEST_REPORT"
echo "  - Executed notebook: $OUTPUT_NOTEBOOK"
echo "  - HTML report: $OUTPUT_HTML"
echo ""
echo "Next steps:"
echo "  1. Open $OUTPUT_HTML in a browser"
echo "  2. Review the prioritized action plan"
echo "  3. Share with your team"
echo "  4. Track actions in the exported CSV"
echo ""

# Extract key metrics from the report (if jq is available)
if command -v jq &> /dev/null; then
    echo "Key Metrics:"
    echo "=============="

    TOTAL_COST=$(jq -r '[.monthly_costs[].cost] | add' "$LATEST_REPORT" 2>/dev/null || echo "N/A")
    if [ "$TOTAL_COST" != "N/A" ]; then
        printf "  Total 3-month cost: \$%'.2f\n" "$TOTAL_COST"
    fi

    IDLE_SAVINGS=$(jq -r '.cloudwatch_analysis.savings // 0' "$LATEST_REPORT" 2>/dev/null || echo "0")
    printf "  Idle instance savings: \$%'.2f/month\n" "$IDLE_SAVINGS"

    EBS_SAVINGS=$(jq -r '.ebs_analysis.total_savings // 0' "$LATEST_REPORT" 2>/dev/null || echo "0")
    printf "  EBS optimization savings: \$%'.2f/month\n" "$EBS_SAVINGS"

    TOTAL_SAVINGS=$(echo "$IDLE_SAVINGS + $EBS_SAVINGS" | bc 2>/dev/null || echo "N/A")
    if [ "$TOTAL_SAVINGS" != "N/A" ]; then
        printf "  Total potential savings: \$%'.2f/month\n" "$TOTAL_SAVINGS"
    fi
    echo ""
fi

echo "To view the report:"
echo "  open $OUTPUT_HTML"
echo ""
echo "To set up weekly automation:"
echo "  crontab -e"
echo "  # Add: 0 9 * * 1 $SCRIPT_DIR/$(basename "$0")"
echo ""

exit 0
