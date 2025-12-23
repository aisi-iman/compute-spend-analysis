# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWS EC2 Cost Optimization Analysis System - a Python-based FinOps tool that analyzes EC2 spending across AWS accounts, identifies optimization opportunities, and generates interactive reports with actionable recommendations.

## Common Commands

```bash
# Install dependencies
uv pip install -r requirements.txt

# Generate cost analysis data (fetches from AWS APIs)
uv run ec2_cost_analyzer.py

# Launch interactive report
uv run jupyter lab ec2_cost_optimization_report.ipynb

# Export notebook to shareable HTML
uv run jupyter nbconvert --to html ec2_cost_optimization_report.ipynb

# Run automated weekly report generation
./automation/run_weekly_report.sh
```

## Architecture

### Data Flow
```
AWS APIs (Cost Explorer, EC2, CloudWatch, Compute Optimizer)
    ↓
ec2_cost_analyzer.py (EC2CostAnalyzer class)
    ↓
JSON Report (data/ec2_cost_report_YYYYMMDD_HHMMSS.json)
    ↓
report_utils.py (visualization utilities)
    ↓
ec2_cost_optimization_report.ipynb (interactive dashboard)
```

### Core Components

**`ec2_cost_analyzer.py`** - Main analysis engine containing the `EC2CostAnalyzer` class:
- Multi-source data collection from 9 AWS services (Cost Explorer, EC2, CloudWatch, Compute Optimizer, EBS, Spot Pricing, RI/Savings Plans, Resource Groups Tagging, STS)
- Multi-region iteration for comprehensive coverage
- Analysis phases: cost collection → utilization metrics → optimization identification → action plan generation
- Optimization detection: idle instances (<5% CPU), underutilization (<20% CPU), orphaned EBS volumes, old snapshots, spot migration opportunities, tag compliance gaps

**`report_utils.py`** - Visualization and utility functions:
- Data loading and DataFrame parsing from JSON reports
- Interactive Plotly charts (time-series, bar charts, pie charts, waterfall)
- AWS CLI command generation for recommended actions
- Batch script generation for automation

**`ec2_cost_optimization_report.ipynb`** - 8-section interactive dashboard:
- Executive Summary, Cost Breakdown, Resource Utilization, Storage Optimization, Commitment Analysis, Action Plan, Recommendations, Export Options

### JSON Report Structure
Generated reports contain: `generated_at`, `time_period`, `monthly_costs`, `instance_type_costs`, `region_costs`, `tag_analysis`, `cloudwatch_analysis`, `ebs_analysis`, `compute_optimizer_analysis`, `spot_analysis`, `ri_sp_analysis`, `action_plan`

## Multi-Account Analysis

Analyze EC2 costs across multiple AWS accounts and generate a combined report.

### Setup

```bash
# Run the setup wizard (one-time)
uv run python setup_multi_account.py

# Follow prompts to enter your SSO URL, authenticate in browser
# This auto-discovers all your accounts and generates config files
```

### How It Works

1. **SSO Login** - You enter your SSO portal URL and authenticate in browser. AWS stores an access token in `~/.aws/sso/cache/`

2. **Account Discovery** - The script calls AWS SSO API `list_accounts` to find all accounts your SSO user can access

3. **Role Discovery** - For each account, it calls `list_account_roles` to find available IAM roles, preferring ones with "Extended" or "Admin" in the name

4. **Config Generation** - Creates two files:
   - `~/.aws/config` - AWS CLI profiles for each account
   - `accounts_config.yaml` - Configuration for the cost analyzer

### Running Multi-Account Analysis

```bash
# Ensure SSO session is active
aws sso login --sso-session cost-analyzer-setup

# Run analysis across all accounts
uv run python multi_account_analyzer.py

# View combined report in notebook
uv run jupyter lab ec2_cost_optimization_report.ipynb
```

### Configuration File

`accounts_config.yaml` controls which accounts are analyzed:

```yaml
accounts:
  - profile: "aisi-production"
    name: "Production"
    account_id: "111111111111"
    enabled: true  # Set to false to skip this account

settings:
  continue_on_error: true  # Continue if one account fails
  include_zero_spend_accounts: false
```

### Multi-Account Report Structure

Combined reports include:
- Per-account breakdowns with individual cost data
- Aggregated totals across all accounts
- Cost-by-account rankings
- Combined fiscal year forecast

## AWS Permissions Required

The analyzer requires IAM permissions for: Cost Explorer (ce:*), EC2 (ec2:Describe*), CloudWatch (cloudwatch:GetMetricStatistics), Compute Optimizer (compute-optimizer:Get*), Resource Groups Tagging (tag:GetResources), STS (sts:GetCallerIdentity)

## Key Dependencies

- **boto3** - AWS SDK for all API interactions
- **pandas** - Data analysis and manipulation
- **plotly** - Interactive visualizations
- **jupyter** - Notebook environment with ipywidgets

## Development guidelines

- Always use uv for running python scripts instead of pip etc
- Find the relevant issue in Linear and update Linear MCP when you have completed a major task. Ask me if you are not sure whether you have completed a major task.
- Update claude.md whenever there has been a major architectural or feature change
- Carefully consider the project structure before creating or editing files. Where there is a file you could sensibly add to, prioritise that over creating a new file. Where you create a new file, consider swe best practices for file structures.