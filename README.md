# EC2 Cost Optimization Report - Usage Guide

## Overview

This reporting solution provides comprehensive, visual analysis of EC2 costs with actionable recommendations for Engineering and DevOps teams.

## What Was Created

### 1. **ec2_cost_optimization_report.ipynb** - Main Jupyter Notebook
   - 8 comprehensive sections covering all cost optimization areas
   - Interactive Plotly visualizations with hover tooltips
   - Actionable AWS CLI commands for each recommendation
   - Exportable to HTML for sharing

### 2. **report_utils.py** - Helper Functions Library
   - Data loading and parsing functions
   - Reusable visualization functions
   - AWS CLI command generators
   - Formatting utilities

### 3. **requirements.txt** - Updated Dependencies
   - All visualization libraries (pandas, plotly, matplotlib, seaborn)
   - Jupyter notebook environment

## Quick Start

### Step 1: Generate Fresh Cost Data
```bash
# Run the cost analyzer to generate latest JSON report
uv run ec2_cost_analyzer.py
```

This will create a file like `ec2_cost_report_YYYYMMDD_HHMMSS.json`

### Step 2: Launch Jupyter Notebook
```bash
# Start Jupyter Lab
uv run jupyter lab ec2_cost_optimization_report.ipynb

# Or use Jupyter Notebook
uv run jupyter notebook ec2_cost_optimization_report.ipynb
```

### Step 3: Update Configuration
In the notebook's configuration cell, update:
```python
REPORT_JSON_PATH = "ec2_cost_report_20251216_172909.json"  # <- Update this
```

### Step 4: Run All Cells
- Click "Kernel" → "Restart & Run All"
- Or press Shift+Enter through each cell

## Report Sections

### Section 1: Executive Summary
- **KPI Cards**: Total cost, average, trends, optimization potential
- **Monthly Trend Chart**: Cost and usage over time with dual axis
- **Savings Breakdown**: Pie chart of opportunities by category

**Use Case**: Quick overview for stakeholders or weekly reviews

### Section 2: Cost Breakdown Analysis
- **Instance Type Costs**: Top 15 most expensive types with horizontal bar chart
- **Regional Distribution**: Pie chart showing cost concentration
- **Tag-Based Analysis**: Breakdowns by Environment, Project, Owner, Component

**Use Case**: Understanding where money is going and identifying outliers

### Section 3: Resource Utilization Analysis
- **Idle Instances** (CPU < 5%): Table with instance IDs, owners, and stop commands
- **Underutilized Instances** (CPU < 20%): Rightsizing candidates
- **High Utilization** (CPU > 80%): Potential bottlenecks needing attention

**Use Case**: Finding waste and performance issues

### Section 4: Storage Optimization
- **Orphaned EBS Volumes**: Unattached volumes with delete commands
- **GP2→GP3 Migration**: Opportunities for 20% savings
- **Old Snapshots**: Snapshots > 1 year old

**Use Case**: Reducing storage costs with immediate actions

### Section 5: Commitment & Pricing Optimization
- **Savings Plans Coverage**: Gauge chart showing current vs target
- **Purchase Recommendations**: Specific SP/RI recommendations with ROI
- **Spot Opportunities**: Eligible instances for spot migration

**Use Case**: Long-term cost reduction through commitments

### Section 6: Prioritized Action Plan
- **Waterfall Chart**: Cumulative savings visualization
- **Action Table**: Ranked by savings with priority, effort, and category
- **Timeline**: Suggested implementation schedule

**Use Case**: Execution roadmap for optimization work

### Section 7: Recommendations Summary
- **Top 10 Actions**: Ranked by savings potential
- **Timeline & Owners**: Suggested schedule and responsibilities

**Use Case**: Final checklist for implementation

### Section 8: Export Options
- **CSV Export**: Action plan for tracking
- **Batch Scripts**: Shell scripts for bulk operations
- **HTML Export**: Command to share report

**Use Case**: Distributing recommendations to teams

## Key Features

### 🎨 Interactive Visualizations
- **Hover tooltips**: See instance IDs, costs, and details
- **Zoom/pan**: Drill into specific time periods or cost ranges
- **Plotly charts**: Professional, publication-ready graphics

### 🛠️ Actionable Commands
Every recommendation includes:
- Specific instance IDs or volume IDs
- Ready-to-run AWS CLI commands
- Console links for manual actions
- Batch scripts for bulk operations

Example:
```bash
aws ec2 stop-instances --instance-ids i-0123456789abcdef --region eu-west-2
```

### 📊 Comprehensive Analysis
- **$1.4M annual** optimization potential identified
- **320+ idle instances** detected
- **31 orphaned volumes** found
- **707 spot-eligible** instances

### 🔄 Reusable & Automated
- Parameterized configuration cell
- Works with any ec2_cost_analyzer.py output
- Can be scheduled for weekly/monthly reports

## Advanced Usage

### Export to HTML for Sharing
```bash
# From terminal
uv run jupyter nbconvert --to html ec2_cost_optimization_report.ipynb

# Opens as: ec2_cost_optimization_report.html
```

### Generate Batch Stop Script
In the notebook, run:
```python
idle_instances = idle_df['instance_id'].tolist()
script = generate_batch_stop_script(idle_instances, 'eu-west-2')
with open('stop_idle_instances.sh', 'w') as f:
    f.write(script)
```

Then execute:
```bash
chmod +x stop_idle_instances.sh
./stop_idle_instances.sh
```

### Schedule Recurring Reports

Create a cron job to generate weekly reports:
```bash
# Edit crontab
crontab -e

# Add line (runs every Monday at 9 AM):
0 9 * * 1 cd /home/ubuntu/compute-spend-analysis && uv run ec2_cost_analyzer.py && uv run jupyter nbconvert --execute --to html ec2_cost_optimization_report.ipynb
```

### Customize Thresholds
Update in configuration cell:
```python
COST_THRESHOLD_IDLE = 5.0        # Lower to catch more instances
COST_THRESHOLD_UNDERUTIL = 20.0  # Adjust based on your standards
TARGET_SP_COVERAGE = 70.0        # Your organization's target
```

## Alternative Tools Comparison

While we built a Jupyter Notebook solution, here are alternatives:

| Tool | Pros | Cons | Best For |
|------|------|------|----------|
| **Jupyter Notebook** ✅ | Code transparency, reproducible, rich visuals | Needs Python environment | DevOps teams, technical analysis |
| **Streamlit** | Web UI, easy deployment, real-time updates | Requires hosting | Non-technical stakeholders |
| **Plotly Dash** | Enterprise dashboards, highly customizable | Complex setup | Production dashboards |
| **AWS QuickSight** | Native AWS, managed service | Cost, less flexible | Executive reporting |
| **Grafana** | Real-time monitoring, alerting | Doesn't use Cost Explorer directly | Operations monitoring |

**Our Recommendation**: Start with Jupyter Notebook (you have it now!), then migrate to Streamlit if you need wider team access via web interface.

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"
```bash
# Install dependencies
uv venv
uv pip install -r requirements.txt
```

### "FileNotFoundError: Cost report file not found"
Update `REPORT_JSON_PATH` in configuration cell to point to your JSON file.

### Visualizations don't appear
Make sure you're running in:
- Jupyter Lab (recommended)
- Jupyter Notebook
- VSCode with Jupyter extension

### "No optimization opportunities identified"
This is actually good news! Your infrastructure is well-optimized. Re-run the cost analyzer to ensure data is current.

## File Structure

```
compute-spend-analysis/
├── ec2_cost_analyzer.py                    # Generates cost data
├── ec2_cost_optimization_report.ipynb      # Main report notebook
├── report_utils.py                         # Helper functions
├── requirements.txt                        # Python dependencies
├── ec2_cost_report_YYYYMMDD_HHMMSS.json   # Cost data (generated)
├── action_plan_YYYYMMDD.csv               # Exported actions (generated)
└── stop_idle_instances_YYYYMMDD.sh        # Batch script (generated)
```

## Best Practices

### For Weekly Reports
1. Run `ec2_cost_analyzer.py` every Monday
2. Execute the notebook
3. Export to HTML
4. Share with team via email or Slack
5. Track completed actions in exported CSV

### For Monthly Reviews
1. Generate report on 1st of month
2. Schedule review meeting with stakeholders
3. Assign action items to owners
4. Re-run at mid-month to track progress
5. Celebrate cost savings achieved!

### For Continuous Optimization
1. Set up cron job for weekly automation
2. Create alerts for idle instances > 7 days
3. Implement auto-shutdown for dev environments
4. Review tag compliance monthly
5. Track savings vs. target KPIs

## Key Insights from Current Data

Based on your latest report:

### Cost Drivers
1. **Metal Instances**: $136K (i4i.metal, c5.metal) - 22% of spend
2. **GPU Instances**: $268K (g6, p4d, p5e) - 43% of spend
3. **Memory-Optimized**: $98K (m5 family) - 16% of spend

### Immediate Opportunities
- **$16K/month**: Stop 320 idle instances
- **$5.7K/month**: Delete 31 orphaned volumes
- **$19K/month**: Purchase Savings Plans
- **$71K/month**: Migrate to Spot instances

### Total Potential: $116K/month ($1.4M/year)

## Getting Help

**For technical issues with the notebook:**
- Check Jupyter logs
- Verify all dependencies installed
- Review error messages in notebook cells

**For AWS optimization questions:**
- Consult AWS Cost Optimization documentation
- Engage AWS Support or TAM
- Review AWS Well-Architected Framework

**For questions about this tool:**
- Review this README
- Check comments in `report_utils.py`
- Examine notebook cell markdown

## Contributing Improvements

Have ideas to enhance the report? Common additions:

1. **Memory utilization** (in addition to CPU)
2. **Network traffic analysis**
3. **Cost anomaly detection**
4. **Budget vs. actual tracking**
5. **Team-specific dashboards**

Feel free to extend `report_utils.py` and add new notebook sections!

## Success Metrics

Track these KPIs monthly:
- ✅ Total EC2 spend (should trend down)
- ✅ % of idle instances (target: < 5%)
- ✅ Savings Plans coverage (target: > 70%)
- ✅ Tag compliance (target: 100%)
- ✅ Orphaned resources (target: 0)
- ✅ Spot instance adoption (target: > 30% for eligible workloads)

---

**Report Generated with Claude Code**
For questions or improvements, contact your FinOps or Platform team.
