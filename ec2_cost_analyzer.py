#!/usr/bin/env python3
"""
EC2 Cost Analysis Script
Analyzes EC2 spending over the last 3 months using AWS Cost Explorer API
"""

import boto3
import json
from datetime import datetime, timedelta
from collections import defaultdict
from decimal import Decimal
import sys


class DecimalEncoder(json.JSONEncoder):
    """Helper to encode Decimal objects to float for JSON serialization"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


class EC2CostAnalyzer:
    def __init__(self, session: boto3.Session = None, account_name: str = None):
        """
        Initialize AWS clients for multi-source analysis.

        Args:
            session: Optional boto3 Session. If None, uses default credentials.
            account_name: Optional friendly name for the account (for logging/reports).
        """
        self.session = session or boto3.Session()
        self.account_name = account_name

        try:
            # Core clients - use session for cross-account support
            self.ce_client = self.session.client('ce')
            self.sts_client = self.session.client('sts')

            # New clients for enhanced analysis
            self.ec2_client = self.session.client('ec2')
            self.cloudwatch_client = self.session.client('cloudwatch')
            self.tagging_client = self.session.client('resourcegroupstaggingapi')

            # Compute Optimizer (may not be available in all regions)
            try:
                self.compute_optimizer_client = self.session.client('compute-optimizer')
            except Exception as e:
                print(f"⚠️  Compute Optimizer not available: {e}")
                self.compute_optimizer_client = None

            # Verify credentials
            identity = self.sts_client.get_caller_identity()
            self.account_id = identity['Account']
            display_name = f" ({self.account_name})" if self.account_name else ""
            print(f"✓ Connected to AWS Account: {self.account_id}{display_name}")
            print(f"✓ User/Role: {identity['Arn']}\n")

            # Get available regions for multi-region operations
            regions_response = self.ec2_client.describe_regions()
            self.all_regions = [r['RegionName'] for r in regions_response['Regions']]

        except Exception as e:
            print(f"✗ Error connecting to AWS: {e}")
            raise RuntimeError(f"Failed to initialize AWS connection: {e}")

    def get_date_range(self):
        """Calculate date range for last 3 months"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    def get_fiscal_year_range(self):
        """
        Calculate April to April fiscal year range for forecasting.
        Returns the start of the current fiscal year and the end of the fiscal year.
        Fiscal year runs April 1 to March 31.
        """
        today = datetime.now().date()
        current_year = today.year
        current_month = today.month

        # Determine the current fiscal year
        if current_month >= 4:  # April onwards = current calendar year's fiscal year
            fiscal_year_start = datetime(current_year, 4, 1).date()
            fiscal_year_end = datetime(current_year + 1, 3, 31).date()
        else:  # Jan-March = previous calendar year's fiscal year
            fiscal_year_start = datetime(current_year - 1, 4, 1).date()
            fiscal_year_end = datetime(current_year, 3, 31).date()

        return fiscal_year_start, fiscal_year_end

    def fetch_ec2_costs_by_time(self, granularity='MONTHLY'):
        """Fetch EC2 costs grouped by time period"""
        start_date, end_date = self.get_date_range()

        print(f"Fetching EC2 costs from {start_date} to {end_date}...")

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity=granularity,
                Metrics=['UnblendedCost', 'UsageQuantity'],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Elastic Compute Cloud - Compute']
                    }
                }
            )
            return response['ResultsByTime']
        except Exception as e:
            print(f"✗ Error fetching cost data: {e}")
            return []

    def fetch_total_aws_costs_by_time(self, granularity='MONTHLY'):
        """Fetch total AWS costs (all services) grouped by time period"""
        start_date, end_date = self.get_date_range()

        print(f"Fetching total AWS costs from {start_date} to {end_date}...")

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity=granularity,
                Metrics=['UnblendedCost']
            )
            return response['ResultsByTime']
        except Exception as e:
            print(f"✗ Error fetching total AWS cost data: {e}")
            return []

    def fetch_costs_by_service(self):
        """Fetch costs grouped by AWS service"""
        start_date, end_date = self.get_date_range()

        print("Fetching cost breakdown by service...")

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            return response['ResultsByTime']
        except Exception as e:
            print(f"✗ Error fetching service breakdown: {e}")
            return []

    def fetch_costs_by_instance_type(self):
        """Fetch EC2 costs grouped by instance type"""
        start_date, end_date = self.get_date_range()

        print("Fetching cost breakdown by instance type...")

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Elastic Compute Cloud - Compute']
                    }
                },
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'INSTANCE_TYPE'
                    }
                ]
            )
            return response['ResultsByTime']
        except Exception as e:
            print(f"✗ Error fetching instance type data: {e}")
            return []

    def fetch_costs_by_instance_type_and_tag(self, tag_key: str = 'Component'):
        """
        Fetch EC2 costs grouped by instance type AND a specific tag.

        This enables stacked bar charts showing instance type costs broken down by tag value.

        Args:
            tag_key: The tag to group by (default: 'Component')

        Returns:
            List of results with costs grouped by instance type and tag
        """
        start_date, end_date = self.get_date_range()

        print(f"Fetching cost breakdown by instance type and {tag_key} tag...")

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Elastic Compute Cloud - Compute']
                    }
                },
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'INSTANCE_TYPE'
                    },
                    {
                        'Type': 'TAG',
                        'Key': tag_key
                    }
                ]
            )
            return response['ResultsByTime']
        except Exception as e:
            print(f"✗ Error fetching instance type by tag data: {e}")
            return []

    def analyze_instance_type_by_tag(self, results_by_type_and_tag, tag_key: str = 'Component'):
        """
        Analyze costs by instance type and tag, returning data suitable for stacked bar charts.

        Args:
            results_by_type_and_tag: Results from fetch_costs_by_instance_type_and_tag
            tag_key: The tag key used in the grouping

        Returns:
            Dict with structure: {instance_type: {tag_value: cost, ...}, ...}
        """
        instance_tag_costs = defaultdict(lambda: defaultdict(float))

        for result in results_by_type_and_tag:
            for group in result.get('Groups', []):
                keys = group['Keys']
                # Keys are [instance_type, tag_value]
                instance_type = keys[0] if keys else 'Unknown'
                tag_value = keys[1] if len(keys) > 1 else '<Untagged>'

                # Handle empty or special tag values
                if not tag_value or tag_value == 'None' or tag_value.startswith('$'):
                    tag_value = '<Untagged>'

                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                instance_tag_costs[instance_type][tag_value] += cost

        # Convert to regular dict and calculate totals
        result = {}
        for instance_type, tag_costs in instance_tag_costs.items():
            result[instance_type] = {
                'by_tag': dict(tag_costs),
                'total': sum(tag_costs.values())
            }

        # Sort by total cost descending
        result = dict(sorted(result.items(), key=lambda x: x[1]['total'], reverse=True))

        return result

    def fetch_costs_by_region(self):
        """Fetch EC2 costs grouped by region"""
        start_date, end_date = self.get_date_range()

        print("Fetching cost breakdown by region...")

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Elastic Compute Cloud - Compute']
                    }
                },
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'REGION'
                    }
                ]
            )
            return response['ResultsByTime']
        except Exception as e:
            print(f"✗ Error fetching region data: {e}")
            return []

    def fetch_costs_by_usage_type(self):
        """Fetch EC2 costs grouped by usage type"""
        start_date, end_date = self.get_date_range()

        print("Fetching cost breakdown by usage type...")

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Elastic Compute Cloud - Compute']
                    }
                },
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'USAGE_TYPE'
                    }
                ]
            )
            return response['ResultsByTime']
        except Exception as e:
            print(f"✗ Error fetching usage type data: {e}")
            return []

    def fetch_available_tag_keys(self):
        """Discover what tag keys are available for EC2"""
        start_date, end_date = self.get_date_range()

        print("Discovering available tags...")

        try:
            response = self.ce_client.get_tags(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Elastic Compute Cloud - Compute']
                    }
                }
            )
            return response.get('Tags', [])
        except Exception as e:
            print(f"✗ Error fetching tag keys: {e}")
            return []

    def fetch_costs_by_tag(self, tag_key):
        """Fetch EC2 costs grouped by specific tag key"""
        start_date, end_date = self.get_date_range()

        print(f"Fetching cost breakdown by tag: {tag_key}...")

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Elastic Compute Cloud - Compute']
                    }
                },
                GroupBy=[
                    {
                        'Type': 'TAG',
                        'Key': tag_key
                    }
                ]
            )
            return response['ResultsByTime']
        except Exception as e:
            print(f"✗ Error fetching costs by tag '{tag_key}': {e}")
            return []

    # ========================================================================
    # NEW DATA FETCHING METHODS FOR ENHANCED ANALYSIS
    # ========================================================================

    def fetch_cloudwatch_metrics(self):
        """Fetch CloudWatch CPU utilization metrics for EC2 instances"""
        print("Fetching CloudWatch utilization metrics...")

        try:
            # Use tagging API to discover EC2 instances
            instances = []
            paginator = self.tagging_client.get_paginator('get_resources')

            for page in paginator.paginate(
                ResourceTypeFilters=['ec2:instance'],
                TagFilters=[]
            ):
                for resource in page['ResourceTagMappingList']:
                    instance_arn = resource['ResourceARN']
                    instance_id = instance_arn.split('/')[-1]

                    # Extract region from ARN
                    region = instance_arn.split(':')[3]

                    tags = {tag['Key']: tag['Value'] for tag in resource.get('Tags', [])}

                    instances.append({
                        'InstanceId': instance_id,
                        'Region': region,
                        'Tags': tags
                    })

            print(f"  Found {len(instances)} instances across all regions")

            # Get CloudWatch metrics for each instance
            for instance in instances:
                try:
                    # Get instance details including type
                    ec2_regional = self.session.client('ec2', region_name=instance['Region'])
                    try:
                        ec2_response = ec2_regional.describe_instances(InstanceIds=[instance['InstanceId']])
                        if ec2_response['Reservations']:
                            inst_details = ec2_response['Reservations'][0]['Instances'][0]
                            instance['InstanceType'] = inst_details.get('InstanceType', 'N/A')
                            instance['State'] = inst_details.get('State', {}).get('Name', 'unknown')
                        else:
                            instance['InstanceType'] = 'N/A'
                            instance['State'] = 'unknown'
                    except Exception as e:
                        instance['InstanceType'] = 'N/A'
                        instance['State'] = 'unknown'

                    # Get CloudWatch metrics
                    cw_regional = self.session.client('cloudwatch', region_name=instance['Region'])

                    # Get average CPU over last 7 days
                    response = cw_regional.get_metric_statistics(
                        Namespace='AWS/EC2',
                        MetricName='CPUUtilization',
                        Dimensions=[{'Name': 'InstanceId', 'Value': instance['InstanceId']}],
                        StartTime=datetime.now() - timedelta(days=7),
                        EndTime=datetime.now(),
                        Period=3600,  # 1 hour
                        Statistics=['Average', 'Maximum']
                    )

                    if response['Datapoints']:
                        instance['AvgCPU'] = sum(dp['Average'] for dp in response['Datapoints']) / len(response['Datapoints'])
                        instance['MaxCPU'] = max(dp['Maximum'] for dp in response['Datapoints'])
                        instance['DatapointCount'] = len(response['Datapoints'])
                    else:
                        instance['AvgCPU'] = None
                        instance['MaxCPU'] = None
                except Exception as e:
                    instance['AvgCPU'] = None
                    instance['MaxCPU'] = None
                    if 'InstanceType' not in instance:
                        instance['InstanceType'] = 'N/A'

            return instances

        except Exception as e:
            print(f"✗ Error fetching CloudWatch metrics: {e}")
            return []

    def fetch_ebs_volumes(self):
        """Fetch EBS volumes and snapshots across all regions"""
        print("Fetching EBS volumes and snapshots...")

        all_volumes = []
        all_snapshots = []

        for region in self.all_regions:
            try:
                ec2_regional = self.session.client('ec2', region_name=region)

                # Get volumes
                volumes_response = ec2_regional.describe_volumes()
                for volume in volumes_response['Volumes']:
                    volume['Region'] = region
                    all_volumes.append(volume)

                # Get snapshots owned by this account
                snapshots_response = ec2_regional.describe_snapshots(OwnerIds=['self'])
                for snapshot in snapshots_response['Snapshots']:
                    snapshot['Region'] = region
                    all_snapshots.append(snapshot)

            except Exception as e:
                print(f"  ⚠️  Could not access region {region}: {e}")

        print(f"  Found {len(all_volumes)} volumes and {len(all_snapshots)} snapshots")
        return {'volumes': all_volumes, 'snapshots': all_snapshots}

    def fetch_compute_optimizer_recommendations(self):
        """Fetch Compute Optimizer rightsizing recommendations"""
        if not self.compute_optimizer_client:
            print("⚠️  Skipping Compute Optimizer (not available)")
            return []

        print("Fetching Compute Optimizer recommendations...")

        try:
            recommendations = []
            paginator = self.compute_optimizer_client.get_paginator('get_ec2_instance_recommendations')

            for page in paginator.paginate():
                recommendations.extend(page.get('instanceRecommendations', []))

            print(f"  Found {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            print(f"✗ Error fetching Compute Optimizer data: {e}")
            print("  Note: Compute Optimizer may need to be enabled in your account")
            return []

    def fetch_spot_pricing(self, instance_types):
        """Fetch current spot prices for given instance types"""
        print("Fetching spot pricing data...")

        spot_prices = {}

        try:
            # Get spot prices for each instance type (sample from one region)
            for instance_type in instance_types[:20]:  # Limit to top 20 types
                try:
                    response = self.ec2_client.describe_spot_price_history(
                        InstanceTypes=[instance_type],
                        ProductDescriptions=['Linux/UNIX'],
                        MaxResults=1,
                        StartTime=datetime.now() - timedelta(hours=1)
                    )

                    if response['SpotPriceHistory']:
                        spot_prices[instance_type] = {
                            'spot_price': float(response['SpotPriceHistory'][0]['SpotPrice']),
                            'availability_zone': response['SpotPriceHistory'][0]['AvailabilityZone']
                        }
                except Exception:
                    pass

            print(f"  Found spot prices for {len(spot_prices)} instance types")
            return spot_prices

        except Exception as e:
            print(f"✗ Error fetching spot pricing: {e}")
            return {}

    def fetch_tag_compliance(self):
        """Fetch comprehensive tag data from Resource Groups Tagging API"""
        print("Fetching tag compliance data...")

        try:
            ec2_resources = []
            ebs_resources = []

            # Get EC2 instances
            paginator = self.tagging_client.get_paginator('get_resources')
            for page in paginator.paginate(ResourceTypeFilters=['ec2:instance']):
                ec2_resources.extend(page['ResourceTagMappingList'])

            # Get EBS volumes
            for page in paginator.paginate(ResourceTypeFilters=['ec2:volume']):
                ebs_resources.extend(page['ResourceTagMappingList'])

            print(f"  Found {len(ec2_resources)} EC2 instances and {len(ebs_resources)} EBS volumes")
            return {'ec2': ec2_resources, 'ebs': ebs_resources}

        except Exception as e:
            print(f"✗ Error fetching tag compliance data: {e}")
            return {'ec2': [], 'ebs': []}

    def fetch_ri_savings_plans_data(self):
        """Fetch Reserved Instance and Savings Plans utilization and recommendations"""
        print("Fetching RI/Savings Plans data...")

        start_date, end_date = self.get_date_range()
        data = {}

        try:
            # Get RI utilization
            try:
                ri_util = self.ce_client.get_reservation_utilization(
                    TimePeriod={'Start': start_date, 'End': end_date},
                    Granularity='MONTHLY'
                )
                data['ri_utilization'] = ri_util.get('UtilizationsByTime', [])
            except Exception as e:
                print(f"  ℹ️  No RI data: {e}")
                data['ri_utilization'] = []

            # Get RI coverage
            try:
                ri_coverage = self.ce_client.get_reservation_coverage(
                    TimePeriod={'Start': start_date, 'End': end_date},
                    Granularity='MONTHLY'
                )
                data['ri_coverage'] = ri_coverage.get('CoveragesByTime', [])
            except Exception as e:
                data['ri_coverage'] = []

            # Get Savings Plans utilization
            try:
                sp_util = self.ce_client.get_savings_plans_utilization(
                    TimePeriod={'Start': start_date, 'End': end_date},
                    Granularity='MONTHLY'
                )
                data['sp_utilization'] = sp_util.get('SavingsPlansUtilizationsByTime', [])
            except Exception as e:
                data['sp_utilization'] = []

            # Get Savings Plans coverage
            try:
                sp_coverage = self.ce_client.get_savings_plans_coverage(
                    TimePeriod={'Start': start_date, 'End': end_date},
                    Granularity='MONTHLY'
                )
                data['sp_coverage'] = sp_coverage.get('SavingsPlansCoverages', [])
            except Exception as e:
                data['sp_coverage'] = []

            # Get purchase recommendations
            try:
                sp_recommendations = self.ce_client.get_savings_plans_purchase_recommendation(
                    SavingsPlansType='COMPUTE_SP',
                    TermInYears='ONE_YEAR',
                    PaymentOption='NO_UPFRONT',
                    LookbackPeriodInDays='THIRTY_DAYS'
                )
                data['sp_recommendations'] = sp_recommendations.get('SavingsPlansPurchaseRecommendation', {})
            except Exception as e:
                print(f"  ℹ️  No SP recommendations: {e}")
                data['sp_recommendations'] = {}

            print(f"  Fetched RI/SP data successfully")
            return data

        except Exception as e:
            print(f"✗ Error fetching RI/SP data: {e}")
            return {}

    def fetch_fiscal_year_forecast(self):
        """
        Fetch actual spend + forecast for fiscal year (April to April).
        Returns monthly breakdown with actual spend for past months and
        forecast for remaining months.
        """
        print("Fetching fiscal year forecast (April to April)...")

        fiscal_start, fiscal_end = self.get_fiscal_year_range()
        today = datetime.now().date()

        # Ensure we don't query future dates for actuals
        actuals_end = min(today, fiscal_end)

        result = {
            'fiscal_year': f"{fiscal_start.strftime('%Y-%m-%d')} to {fiscal_end.strftime('%Y-%m-%d')}",
            'fiscal_year_label': f"FY{fiscal_start.year}/{fiscal_end.year}",
            'generated_at': datetime.now().isoformat(),
            'monthly_data': [],
            'total_actual': 0,
            'total_forecast': 0,
            'total_projected': 0
        }

        try:
            # PART 1: Fetch actual costs for completed months (from fiscal year start to today)
            if fiscal_start < actuals_end:
                print(f"  Fetching actual costs from {fiscal_start} to {actuals_end}...")

                # Fetch ALL account costs (not just EC2)
                response = self.ce_client.get_cost_and_usage(
                    TimePeriod={
                        'Start': fiscal_start.strftime('%Y-%m-%d'),
                        'End': actuals_end.strftime('%Y-%m-%d')
                    },
                    Granularity='MONTHLY',
                    Metrics=['UnblendedCost']
                )

                for period in response.get('ResultsByTime', []):
                    period_start = period['TimePeriod']['Start']
                    period_end = period['TimePeriod']['End']
                    cost = float(period['Total']['UnblendedCost']['Amount'])

                    result['monthly_data'].append({
                        'period_start': period_start,
                        'period_end': period_end,
                        'month_label': datetime.strptime(period_start, '%Y-%m-%d').strftime('%b %Y'),
                        'cost': cost,
                        'type': 'actual'
                    })
                    result['total_actual'] += cost

                print(f"    Found {len(response.get('ResultsByTime', []))} months of actual data")

            # PART 2: Fetch forecast for remaining months (from today to fiscal year end)
            # Cost Explorer forecast requires at least 1 day in the future
            forecast_start = today + timedelta(days=1)

            if forecast_start < fiscal_end:
                print(f"  Fetching forecast from {forecast_start} to {fiscal_end}...")

                try:
                    forecast_response = self.ce_client.get_cost_forecast(
                        TimePeriod={
                            'Start': forecast_start.strftime('%Y-%m-%d'),
                            'End': fiscal_end.strftime('%Y-%m-%d')
                        },
                        Metric='UNBLENDED_COST',
                        Granularity='MONTHLY'
                    )

                    # Process forecast results
                    for period in forecast_response.get('ForecastResultsByTime', []):
                        period_start = period['TimePeriod']['Start']
                        period_end = period['TimePeriod']['End']
                        mean_value = float(period.get('MeanValue', 0))

                        # Get prediction intervals if available
                        prediction_low = None
                        prediction_high = None
                        if 'PredictionIntervalLowerBound' in period:
                            prediction_low = float(period['PredictionIntervalLowerBound'])
                        if 'PredictionIntervalUpperBound' in period:
                            prediction_high = float(period['PredictionIntervalUpperBound'])

                        result['monthly_data'].append({
                            'period_start': period_start,
                            'period_end': period_end,
                            'month_label': datetime.strptime(period_start, '%Y-%m-%d').strftime('%b %Y'),
                            'cost': mean_value,
                            'type': 'forecast',
                            'prediction_low': prediction_low,
                            'prediction_high': prediction_high
                        })
                        result['total_forecast'] += mean_value

                    # Also capture the total forecast
                    if 'Total' in forecast_response:
                        result['forecast_total_from_api'] = {
                            'mean': float(forecast_response['Total'].get('Amount', 0)),
                            'unit': forecast_response['Total'].get('Unit', 'USD')
                        }

                    print(f"    Generated forecast for {len(forecast_response.get('ForecastResultsByTime', []))} months")

                except self.ce_client.exceptions.DataUnavailableException as e:
                    print(f"  ⚠️  Forecast not available: {e}")
                    print("      (Forecast requires sufficient historical data)")
                except Exception as e:
                    print(f"  ⚠️  Could not generate forecast: {e}")

            # Calculate total projected spend
            result['total_projected'] = result['total_actual'] + result['total_forecast']

            # Sort monthly data by period
            result['monthly_data'].sort(key=lambda x: x['period_start'])

            print(f"\n  📊 Fiscal Year Summary ({result['fiscal_year_label']}):")
            print(f"     Actual spend to date: ${result['total_actual']:,.2f}")
            print(f"     Forecasted remaining: ${result['total_forecast']:,.2f}")
            print(f"     Total projected:      ${result['total_projected']:,.2f}")

            return result

        except Exception as e:
            print(f"✗ Error fetching fiscal year forecast: {e}")
            return result

    def analyze_fiscal_year_forecast(self, forecast_data):
        """Analyze and display fiscal year forecast data"""
        print("\n" + "="*80)
        print("FISCAL YEAR FORECAST (APRIL TO APRIL)")
        print("="*80)

        if not forecast_data or not forecast_data.get('monthly_data'):
            print("\n⚠️  No fiscal year forecast data available")
            return forecast_data

        fy_label = forecast_data.get('fiscal_year_label', 'FY')
        monthly_data = forecast_data.get('monthly_data', [])

        print(f"\n📅 {fy_label}: {forecast_data.get('fiscal_year', 'N/A')}")
        print(f"\n{'Month':<12} {'Type':<10} {'Cost':>15}")
        print("-" * 40)

        for month in monthly_data:
            month_label = month.get('month_label', 'N/A')
            cost_type = month.get('type', 'unknown')
            cost = month.get('cost', 0)
            type_indicator = "📊" if cost_type == 'actual' else "🔮"
            print(f"{month_label:<12} {type_indicator} {cost_type:<8} ${cost:>12,.2f}")

        print("-" * 40)
        print(f"\n💰 TOTALS:")
        print(f"   Actual spend (YTD):     ${forecast_data.get('total_actual', 0):>12,.2f}")
        print(f"   Forecasted remaining:   ${forecast_data.get('total_forecast', 0):>12,.2f}")
        print(f"   ─────────────────────────────────────")
        print(f"   Projected FY Total:     ${forecast_data.get('total_projected', 0):>12,.2f}")

        # Calculate average monthly
        actual_months = len([m for m in monthly_data if m.get('type') == 'actual'])
        forecast_months = len([m for m in monthly_data if m.get('type') == 'forecast'])

        if actual_months > 0:
            avg_actual = forecast_data.get('total_actual', 0) / actual_months
            print(f"\n📈 Average monthly spend (actual): ${avg_actual:,.2f}")

        if forecast_months > 0:
            avg_forecast = forecast_data.get('total_forecast', 0) / forecast_months
            print(f"📈 Average monthly forecast:       ${avg_forecast:,.2f}")

        return forecast_data

    def analyze_time_trends(self, results_by_time):
        """Analyze monthly cost trends"""
        print("\n" + "="*80)
        print("MONTHLY COST TRENDS")
        print("="*80)

        monthly_costs = []
        for result in results_by_time:
            period = result['TimePeriod']
            cost = float(result['Total']['UnblendedCost']['Amount'])
            usage = float(result['Total']['UsageQuantity']['Amount'])

            monthly_costs.append({
                'period': f"{period['Start']} to {period['End']}",
                'cost': cost,
                'usage': usage
            })

            print(f"\nPeriod: {period['Start']} to {period['End']}")
            print(f"  Cost: ${cost:,.2f}")
            print(f"  Usage: {usage:,.2f} hours")

        # Calculate trends
        if len(monthly_costs) >= 2:
            first_month = monthly_costs[0]['cost']
            last_month = monthly_costs[-1]['cost']

            if first_month > 0:
                change_pct = ((last_month - first_month) / first_month) * 100
                print(f"\n📊 Trend: {change_pct:+.1f}% change from first to last month")

            total_cost = sum(m['cost'] for m in monthly_costs)
            avg_cost = total_cost / len(monthly_costs)
            print(f"📊 Total 3-month cost: ${total_cost:,.2f}")
            print(f"📊 Average monthly cost: ${avg_cost:,.2f}")

        return monthly_costs

    def analyze_total_aws_costs(self, results_by_time):
        """Analyze total AWS costs (all services) by time period"""
        print("\n" + "="*80)
        print("TOTAL AWS COSTS (ALL SERVICES)")
        print("="*80)

        monthly_costs = []
        for result in results_by_time:
            period = result['TimePeriod']
            cost = float(result['Total']['UnblendedCost']['Amount'])

            monthly_costs.append({
                'period': f"{period['Start']} to {period['End']}",
                'cost': cost
            })

            print(f"\nPeriod: {period['Start']} to {period['End']}")
            print(f"  Total AWS Cost: ${cost:,.2f}")

        if len(monthly_costs) >= 2:
            total_cost = sum(m['cost'] for m in monthly_costs)
            avg_cost = total_cost / len(monthly_costs)
            print(f"\n📊 Total 3-month AWS cost: ${total_cost:,.2f}")
            print(f"📊 Average monthly AWS cost: ${avg_cost:,.2f}")

        return monthly_costs

    def analyze_service_costs(self, results_by_service):
        """Analyze costs by AWS service"""
        print("\n" + "="*80)
        print("COST BREAKDOWN BY AWS SERVICE")
        print("="*80)

        # Aggregate costs across all time periods
        service_costs = defaultdict(float)

        for result in results_by_service:
            for group in result.get('Groups', []):
                service = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                service_costs[service] += cost

        # Sort by cost descending
        sorted_services = sorted(service_costs.items(), key=lambda x: x[1], reverse=True)

        total_cost = sum(service_costs.values())

        print(f"\nAWS Services by Cost (Total: ${total_cost:,.2f}):\n")

        for i, (service, cost) in enumerate(sorted_services[:15], 1):
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            bar = "█" * int(percentage / 2)
            # Truncate long service names
            service_short = service[:40] + "..." if len(service) > 40 else service
            print(f"{i:2d}. {service_short:45s} ${cost:12,.2f} ({percentage:5.1f}%) {bar}")

        if len(sorted_services) > 15:
            other_cost = sum(cost for _, cost in sorted_services[15:])
            other_pct = (other_cost / total_cost * 100) if total_cost > 0 else 0
            print(f"    {'Others':45s} ${other_cost:12,.2f} ({other_pct:5.1f}%)")

        return dict(sorted_services)

    def analyze_instance_types(self, results_by_instance_type):
        """Analyze costs by instance type"""
        print("\n" + "="*80)
        print("COST BREAKDOWN BY INSTANCE TYPE")
        print("="*80)

        # Aggregate costs across all time periods
        instance_costs = defaultdict(float)

        for result in results_by_instance_type:
            for group in result.get('Groups', []):
                instance_type = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                instance_costs[instance_type] += cost

        # Sort by cost descending
        sorted_instances = sorted(instance_costs.items(), key=lambda x: x[1], reverse=True)

        total_cost = sum(instance_costs.values())

        print(f"\nTop Instance Types by Cost (Total: ${total_cost:,.2f}):\n")

        for i, (instance_type, cost) in enumerate(sorted_instances[:15], 1):
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"{i:2d}. {instance_type:20s} ${cost:10,.2f} ({percentage:5.1f}%) {bar}")

        if len(sorted_instances) > 15:
            other_cost = sum(cost for _, cost in sorted_instances[15:])
            other_pct = (other_cost / total_cost * 100) if total_cost > 0 else 0
            print(f"    {'Others':20s} ${other_cost:10,.2f} ({other_pct:5.1f}%)")

        return dict(sorted_instances)

    def analyze_regions(self, results_by_region):
        """Analyze costs by AWS region"""
        print("\n" + "="*80)
        print("COST BREAKDOWN BY REGION")
        print("="*80)

        # Aggregate costs across all time periods
        region_costs = defaultdict(float)

        for result in results_by_region:
            for group in result.get('Groups', []):
                region = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                region_costs[region] += cost

        # Sort by cost descending
        sorted_regions = sorted(region_costs.items(), key=lambda x: x[1], reverse=True)

        total_cost = sum(region_costs.values())

        print(f"\nRegions by Cost (Total: ${total_cost:,.2f}):\n")

        for i, (region, cost) in enumerate(sorted_regions, 1):
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"{i:2d}. {region:25s} ${cost:10,.2f} ({percentage:5.1f}%) {bar}")

        return dict(sorted_regions)

    def analyze_usage_types(self, results_by_usage_type):
        """Analyze costs by usage type"""
        print("\n" + "="*80)
        print("COST BREAKDOWN BY USAGE TYPE")
        print("="*80)

        # Aggregate costs across all time periods
        usage_costs = defaultdict(float)

        for result in results_by_usage_type:
            for group in result.get('Groups', []):
                usage_type = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                usage_costs[usage_type] += cost

        # Sort by cost descending
        sorted_usage = sorted(usage_costs.items(), key=lambda x: x[1], reverse=True)

        total_cost = sum(usage_costs.values())

        print(f"\nTop Usage Types by Cost (Total: ${total_cost:,.2f}):\n")

        for i, (usage_type, cost) in enumerate(sorted_usage[:15], 1):
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"{i:2d}. {usage_type:45s} ${cost:10,.2f} ({percentage:5.1f}%) {bar}")

        if len(sorted_usage) > 15:
            other_cost = sum(cost for _, cost in sorted_usage[15:])
            other_pct = (other_cost / total_cost * 100) if total_cost > 0 else 0
            print(f"    {'Others':45s} ${other_cost:10,.2f} ({other_pct:5.1f}%)")

        return dict(sorted_usage)

    def analyze_tags(self, tag_key, results_by_tag):
        """Analyze costs by specific tag"""
        print("\n" + "="*80)
        print(f"COST BREAKDOWN BY TAG: {tag_key}")
        print("="*80)

        # Aggregate costs across all time periods
        tag_costs = defaultdict(float)
        untagged_cost = 0

        for result in results_by_tag:
            for group in result.get('Groups', []):
                tag_value = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])

                # Handle untagged resources
                if tag_value == '' or tag_value == 'None' or tag_value.startswith('$'):
                    untagged_cost += cost
                else:
                    tag_costs[tag_value] += cost

        # Sort by cost descending
        sorted_tags = sorted(tag_costs.items(), key=lambda x: x[1], reverse=True)

        total_cost = sum(tag_costs.values()) + untagged_cost

        if total_cost == 0:
            print(f"\n⚠️  No cost data found for tag: {tag_key}")
            return {}

        print(f"\nBreakdown (Total: ${total_cost:,.2f}):\n")

        for i, (tag_value, cost) in enumerate(sorted_tags[:20], 1):
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            bar = "█" * int(percentage / 2)
            display_value = tag_value[:40] if len(tag_value) <= 40 else tag_value[:37] + "..."
            print(f"{i:2d}. {display_value:40s} ${cost:10,.2f} ({percentage:5.1f}%) {bar}")

        if untagged_cost > 0:
            untagged_pct = (untagged_cost / total_cost * 100)
            bar = "█" * int(untagged_pct / 2)
            print(f"    {'<UNTAGGED>':40s} ${untagged_cost:10,.2f} ({untagged_pct:5.1f}%) {bar}")

        if len(sorted_tags) > 20:
            other_cost = sum(cost for _, cost in sorted_tags[20:])
            other_pct = (other_cost / total_cost * 100) if total_cost > 0 else 0
            print(f"    {'<Others>':40s} ${other_cost:10,.2f} ({other_pct:5.1f}%)")

        # Show tagging coverage
        tagged_cost = sum(tag_costs.values())
        tagging_coverage = (tagged_cost / total_cost * 100) if total_cost > 0 else 0

        print(f"\n📊 Tagging Coverage: {tagging_coverage:.1f}% of costs are tagged")
        if tagging_coverage < 80:
            print(f"⚠️  Low tagging coverage - {100-tagging_coverage:.1f}% of costs are untagged")

        return dict(sorted_tags)

    # ========================================================================
    # NEW ANALYSIS METHODS FOR ENHANCED INSIGHTS
    # ========================================================================

    def analyze_cloudwatch_utilization(self, cw_data, instance_costs):
        """Analyze CloudWatch CPU metrics to find idle/underutilized instances"""
        print("\n" + "="*80)
        print("CLOUDWATCH UTILIZATION ANALYSIS")
        print("="*80)

        idle_instances = []
        underutilized_instances = []
        high_utilization_instances = []

        idle_threshold = 5.0
        underutil_threshold = 20.0

        for instance in cw_data:
            if instance.get('AvgCPU') is not None:
                if instance['AvgCPU'] < idle_threshold:
                    idle_instances.append(instance)
                elif instance['AvgCPU'] < underutil_threshold:
                    underutilized_instances.append(instance)
                elif instance['AvgCPU'] > 80:
                    high_utilization_instances.append(instance)

        # Estimate savings
        idle_savings = len(idle_instances) * 50  # Conservative estimate
        underutil_savings = len(underutilized_instances) * 30

        if idle_instances:
            print(f"\n⚠️  IDLE INSTANCES (CPU < {idle_threshold}%): {len(idle_instances)} instances")
            print(f"    Potential monthly savings: ${idle_savings:,.2f}\n")
            for inst in idle_instances[:10]:
                name = inst['Tags'].get('Name', 'N/A')
                print(f"    {inst['InstanceId']:20s} CPU: {inst['AvgCPU']:5.1f}% | {name[:40]}")
            if len(idle_instances) > 10:
                print(f"    ... and {len(idle_instances) - 10} more")

        if underutilized_instances:
            print(f"\n💡 UNDERUTILIZED INSTANCES (CPU < {underutil_threshold}%): {len(underutilized_instances)} instances")
            print(f"    Potential monthly savings from rightsizing: ${underutil_savings:,.2f}\n")
            for inst in underutilized_instances[:10]:
                name = inst['Tags'].get('Name', 'N/A')
                print(f"    {inst['InstanceId']:20s} CPU: {inst['AvgCPU']:5.1f}% | {name[:40]}")
            if len(underutilized_instances) > 10:
                print(f"    ... and {len(underutilized_instances) - 10} more")

        if high_utilization_instances:
            print(f"\n⚠️  HIGH UTILIZATION (CPU > 80%): {len(high_utilization_instances)} instances")
            print(f"    May need upsize for performance\n")
            for inst in high_utilization_instances[:5]:
                name = inst['Tags'].get('Name', 'N/A')
                print(f"    {inst['InstanceId']:20s} CPU: {inst['AvgCPU']:5.1f}% | {name[:40]}")

        return {
            'idle': idle_instances,
            'underutilized': underutilized_instances,
            'high_utilization': high_utilization_instances,
            'savings': idle_savings + underutil_savings
        }

    def analyze_ebs_optimization(self, ebs_data):
        """Analyze EBS volumes for optimization opportunities"""
        print("\n" + "="*80)
        print("EBS STORAGE OPTIMIZATION")
        print("="*80)

        volumes = ebs_data.get('volumes', [])
        snapshots = ebs_data.get('snapshots', [])

        unattached = [v for v in volumes if v['State'] == 'available']
        unattached_size = sum(v['Size'] for v in unattached)
        unattached_cost = unattached_size * 0.10  # $0.10/GB/month for gp3

        # Analyze volume types
        gp2_volumes = [v for v in volumes if v['VolumeType'] == 'gp2']
        gp2_size = sum(v['Size'] for v in gp2_volumes)
        gp2_to_gp3_savings = gp2_size * 0.02  # ~20% savings

        # Old snapshots (> 1 year)
        old_snapshots = []
        for snap in snapshots:
            age_days = (datetime.now(snap['StartTime'].tzinfo) - snap['StartTime']).days
            if age_days > 365:
                old_snapshots.append(snap)

        print(f"\n✓ Total volumes: {len(volumes)} ({sum(v['Size'] for v in volumes):,} GB)")
        print(f"✓ Total snapshots: {len(snapshots)}")

        if unattached:
            print(f"\n⚠️  ORPHANED VOLUMES: {len(unattached)} unattached volumes ({unattached_size:,} GB)")
            print(f"    Monthly waste: ${unattached_cost:,.2f}")
            print(f"    Action: Delete after confirming not needed\n")
            for vol in unattached[:15]:
                vol_name = next((tag['Value'] for tag in vol.get('Tags', []) if tag['Key'] == 'Name'), 'N/A')
                print(f"    {vol['VolumeId']:24s} {vol['Size']:4d} GB {vol['VolumeType']:8s} | {vol['Region']} | {vol_name[:30]}")

        if gp2_volumes:
            print(f"\n💡 GP2 TO GP3 MIGRATION: {len(gp2_volumes)} gp2 volumes ({gp2_size:,} GB)")
            print(f"    Potential monthly savings: ${gp2_to_gp3_savings:,.2f}")
            print(f"    Action: Migrate to gp3 for ~20% cost reduction")

        if old_snapshots:
            print(f"\n💡 OLD SNAPSHOTS: {len(old_snapshots)} snapshots > 1 year old")
            print(f"    Consider lifecycle policy or deletion")

        return {
            'unattached_count': len(unattached),
            'unattached_savings': unattached_cost,
            'gp2_migration_savings': gp2_to_gp3_savings,
            'old_snapshots': len(old_snapshots),
            'total_savings': unattached_cost + gp2_to_gp3_savings
        }

    def analyze_compute_optimizer(self, recommendations):
        """Analyze Compute Optimizer recommendations"""
        print("\n" + "="*80)
        print("COMPUTE OPTIMIZER RECOMMENDATIONS")
        print("="*80)

        if not recommendations:
            print("\n⚠️  No Compute Optimizer recommendations available")
            print("   Enable Compute Optimizer in AWS Console for ML-based rightsizing")
            return {'total_savings': 0, 'recommendations': []}

        # Group by recommendation type
        downsize = []
        upsize = []
        change_family = []

        for rec in recommendations:
            current_type = rec.get('currentInstanceType', 'Unknown')
            recommended_options = rec.get('recommendationOptions', [])

            if recommended_options:
                recommended_type = recommended_options[0].get('instanceType', 'Unknown')

                if current_type > recommended_type:
                    downsize.append(rec)
                elif current_type < recommended_type:
                    upsize.append(rec)
                else:
                    change_family.append(rec)

        total_savings = sum(
            float(opt.get('estimatedMonthlySavings', {}).get('value', 0))
            for rec in recommendations
            for opt in rec.get('recommendationOptions', [])
        ) / len(recommendations) if recommendations else 0

        print(f"\n✓ Found {len(recommendations)} optimization recommendations")
        print(f"💰 Estimated monthly savings potential: ${total_savings:,.2f}\n")

        print(f"📊 Breakdown:")
        print(f"   Downsize opportunities: {len(downsize)}")
        print(f"   Upsize recommendations: {len(upsize)}")
        print(f"   Change instance family: {len(change_family)}")

        # Show top 10 recommendations by savings
        print(f"\n🎯 Top Recommendations:\n")
        sorted_recs = sorted(
            recommendations,
            key=lambda r: float(r.get('recommendationOptions', [{}])[0].get('estimatedMonthlySavings', {}).get('value', 0)),
            reverse=True
        )[:10]

        for i, rec in enumerate(sorted_recs, 1):
            instance_arn = rec.get('instanceArn', '')
            instance_id = instance_arn.split('/')[-1] if '/' in instance_arn else 'Unknown'
            current_type = rec.get('currentInstanceType', 'Unknown')

            if rec.get('recommendationOptions'):
                opt = rec['recommendationOptions'][0]
                rec_type = opt.get('instanceType', 'Unknown')
                savings = float(opt.get('estimatedMonthlySavings', {}).get('value', 0))
                print(f"{i:2d}. {instance_id:20s} {current_type:15s} → {rec_type:15s} | Save: ${savings:6.2f}/mo")

        return {
            'total_savings': total_savings,
            'recommendations': recommendations,
            'count': len(recommendations)
        }

    def analyze_spot_opportunities(self, spot_prices, instance_costs, cw_data):
        """Analyze spot instance opportunities"""
        print("\n" + "="*80)
        print("SPOT INSTANCE OPPORTUNITIES")
        print("="*80)

        if not spot_prices:
            print("\n⚠️  No spot pricing data available")
            return {'savings': 0}

        # Identify spot-eligible workloads based on tags and CPU usage
        spot_eligible = []

        for instance in cw_data:
            component = instance['Tags'].get('Component', '')
            project = instance['Tags'].get('Project', '')
            avg_cpu = instance.get('AvgCPU', 100)
            # Handle None values for avg_cpu
            if avg_cpu is None:
                avg_cpu = 100

            # Custom eligibility logic for single prod account
            is_eligible = (
                'dev-vm' in component.lower() or
                'sandbox' in component.lower() or
                'sandbox' in project.lower() or
                avg_cpu < 20  # Very low utilization
            )

            if is_eligible:
                spot_eligible.append(instance)

        print(f"\n💡 Found {len(spot_eligible)} instances eligible for Spot")
        print(f"   Spot instances offer up to 90% savings for fault-tolerant workloads\n")

        # Show spot prices for common types
        print("Current Spot Prices (sample):\n")
        for inst_type, price_data in list(spot_prices.items())[:10]:
            spot_price = price_data['spot_price']
            # Rough on-demand estimates
            od_prices = {'m5.xlarge': 0.192, 'm5.2xlarge': 0.384, 'm5.4xlarge': 0.768,
                        'c5.xlarge': 0.17, 'c5.2xlarge': 0.34}
            od_price = od_prices.get(inst_type, spot_price * 3)  # Rough estimate
            savings_pct = ((od_price - spot_price) / od_price * 100) if od_price > 0 else 0
            print(f"  {inst_type:15s} Spot: ${spot_price:.4f}/hr | Savings: ~{savings_pct:.0f}%")

        # Estimate potential savings (conservative)
        estimated_savings = len(spot_eligible) * 100  # $100/month per instance

        if spot_eligible:
            print(f"\n🎯 Spot-Eligible Instances:\n")
            for inst in spot_eligible[:10]:
                name = inst['Tags'].get('Name', 'N/A')
                component = inst['Tags'].get('Component', 'N/A')
                print(f"    {inst['InstanceId']:20s} {component:20s} | {name[:30]}")

            print(f"\n💰 Estimated monthly savings: ${estimated_savings:,.2f}")
            print(f"    (Assuming ~70% savings for spot-eligible workloads)")

        return {
            'eligible_count': len(spot_eligible),
            'savings': estimated_savings,
            'eligible_instances': spot_eligible
        }

    def analyze_tag_compliance_detailed(self, tag_data):
        """Analyze tag compliance using Resource Groups Tagging API"""
        print("\n" + "="*80)
        print("TAG COMPLIANCE ANALYSIS (DETAILED)")
        print("="*80)

        ec2_resources = tag_data.get('ec2', [])
        ebs_resources = tag_data.get('ebs', [])

        required_tags = ['owner', 'Project', 'Component']
        ec2_untagged = []
        ec2_missing_required = defaultdict(list)

        for resource in ec2_resources:
            tags = {tag['Key']: tag['Value'] for tag in resource.get('Tags', [])}

            if not tags:
                ec2_untagged.append(resource)
            else:
                for req_tag in required_tags:
                    if req_tag not in tags or not tags[req_tag]:
                        ec2_missing_required[req_tag].append(resource)

        compliance_score = ((len(ec2_resources) - len(ec2_untagged)) / len(ec2_resources) * 100) if ec2_resources else 0

        print(f"\n📊 EC2 Tag Compliance: {compliance_score:.1f}%")
        print(f"   Total EC2 instances: {len(ec2_resources)}")
        print(f"   Untagged instances: {len(ec2_untagged)}")

        for tag, resources in ec2_missing_required.items():
            print(f"   Missing '{tag}' tag: {len(resources)}")

        if ec2_untagged:
            print(f"\n⚠️  UNTAGGED EC2 INSTANCES:\n")
            for resource in ec2_untagged[:15]:
                arn = resource['ResourceARN']
                instance_id = arn.split('/')[-1]
                print(f"    {instance_id}")

        return {
            'compliance_score': compliance_score,
            'untagged_ec2': len(ec2_untagged),
            'missing_tags': dict(ec2_missing_required)
        }

    def analyze_ri_savings_plans(self, ri_sp_data):
        """Analyze RI and Savings Plans utilization and recommendations"""
        print("\n" + "="*80)
        print("RESERVED INSTANCES & SAVINGS PLANS")
        print("="*80)

        sp_coverage_data = ri_sp_data.get('sp_coverage', [])
        sp_util_data = ri_sp_data.get('sp_utilization', [])
        sp_recommendations = ri_sp_data.get('sp_recommendations', {})

        # Calculate average coverage
        if sp_coverage_data:
            latest_coverage = sp_coverage_data[-1] if sp_coverage_data else {}
            coverage_pct = float(latest_coverage.get('Coverage', {}).get('CoverageHours', {}).get('CoverageHoursPercentage', 0))
            print(f"\n💰 Savings Plans Coverage: {coverage_pct:.1f}%")
            print(f"    ({100-coverage_pct:.1f}% of usage not covered by commitments)")
        else:
            print(f"\n💰 Savings Plans Coverage: 0%")
            print(f"    No active Savings Plans detected")

        # Utilization
        if sp_util_data:
            latest_util = sp_util_data[-1] if sp_util_data else {}
            util_pct = float(latest_util.get('Utilization', {}).get('UtilizationPercentage', 0))
            print(f"📊 Savings Plans Utilization: {util_pct:.1f}%")
            if util_pct < 90:
                print(f"    ⚠️  Low utilization - you may be over-committed")
        else:
            print(f"📊 Savings Plans Utilization: N/A")

        # Recommendations
        if sp_recommendations:
            recommendation_details = sp_recommendations.get('SavingsPlansPurchaseRecommendationDetails', [])

            if recommendation_details:
                print(f"\n🎯 PURCHASE RECOMMENDATIONS:\n")

                for i, rec in enumerate(recommendation_details[:5], 1):
                    hourly_commit = float(rec.get('HourlyCommitmentToPurchase', 0))
                    estimated_savings = float(rec.get('EstimatedMonthlySavingsAmount', 0))
                    sp_type = rec.get('SavingsPlansType', 'COMPUTE_SP')

                    print(f"{i}. {sp_type}")
                    print(f"     Hourly commitment: ${hourly_commit:.4f}/hour")
                    print(f"     Estimated monthly savings: ${estimated_savings:,.2f}")
                    print(f"     Estimated ROI: {rec.get('EstimatedROI', 'N/A')}%\n")

                total_potential_savings = sum(float(r.get('EstimatedMonthlySavingsAmount', 0)) for r in recommendation_details)
                print(f"💰 Total potential monthly savings from Savings Plans: ${total_potential_savings:,.2f}")

                return {
                    'coverage': coverage_pct if sp_coverage_data else 0,
                    'utilization': util_pct if sp_util_data else 0,
                    'potential_savings': total_potential_savings
                }
            else:
                print(f"\nℹ️  No Savings Plans purchase recommendations available")
                print(f"   This may indicate good existing coverage or insufficient usage data")
        else:
            print(f"\nℹ️  No Savings Plans recommendations data")

        return {
            'coverage': coverage_pct if sp_coverage_data else 0,
            'potential_savings': 0
        }

    def generate_insights(self, monthly_costs, instance_costs, region_costs, usage_costs):
        """Generate insights and recommendations"""
        print("\n" + "="*80)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("="*80)

        insights = []

        # Cost trend insights
        if len(monthly_costs) >= 2:
            first_month = monthly_costs[0]['cost']
            last_month = monthly_costs[-1]['cost']

            if last_month > first_month * 1.2:
                insights.append(f"⚠️  Spending increased by {((last_month - first_month) / first_month * 100):.1f}% - investigate usage growth")
            elif last_month < first_month * 0.8:
                insights.append(f"✓ Spending decreased by {((first_month - last_month) / first_month * 100):.1f}% - good cost optimization")

        # Instance type insights
        if instance_costs:
            top_instance = max(instance_costs.items(), key=lambda x: x[1])
            total_instance_cost = sum(instance_costs.values())
            top_pct = (top_instance[1] / total_instance_cost * 100)

            if top_pct > 50:
                insights.append(f"🔍 {top_instance[0]} accounts for {top_pct:.1f}% of costs - consider if this is optimal")

            # Check for expensive instance types
            expensive_families = ['p3', 'p4', 'g4', 'g5', 'x1', 'x2', 'z1d']
            for instance_type, cost in instance_costs.items():
                family = instance_type.split('.')[0] if '.' in instance_type else instance_type
                if any(family.startswith(exp) for exp in expensive_families) and cost > 100:
                    insights.append(f"💰 Using expensive instance type {instance_type} (${cost:,.2f}) - verify necessity")

        # Region insights
        if len(region_costs) > 1:
            insights.append(f"🌍 Using {len(region_costs)} regions - consider consolidation for cost savings")

        # Usage type insights
        if usage_costs:
            # Check for spot vs on-demand
            spot_cost = sum(cost for usage_type, cost in usage_costs.items() if 'Spot' in usage_type or 'SpotUsage' in usage_type)
            total_usage_cost = sum(usage_costs.values())

            if total_usage_cost > 0:
                spot_pct = (spot_cost / total_usage_cost * 100)
                if spot_pct < 10 and total_usage_cost > 1000:
                    insights.append(f"💡 Only {spot_pct:.1f}% using Spot instances - consider Spot for non-critical workloads")

        print()
        if insights:
            for insight in insights:
                print(f"  {insight}")
        else:
            print("  ✓ No major cost anomalies detected")

        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        print("""
  1. Review instance sizing - right-size over-provisioned instances
  2. Consider Reserved Instances or Savings Plans for steady-state workloads
  3. Implement auto-scaling to match capacity with demand
  4. Use Spot Instances for fault-tolerant workloads (up to 90% savings)
  5. Schedule non-production instances to run only during business hours
  6. Review and terminate unused/idle instances
  7. Consider newer generation instances for better price/performance
  8. Enable AWS Cost Anomaly Detection for proactive alerts
        """)

    def generate_action_plan(self, all_optimization_data):
        """Generate prioritized action plan with savings and AWS CLI commands"""
        print("\n" + "="*80)
        print("🎯 PRIORITIZED ACTION PLAN")
        print("="*80)

        cw_data = all_optimization_data.get('cloudwatch', {})
        ebs_data = all_optimization_data.get('ebs', {})
        compute_opt_data = all_optimization_data.get('compute_optimizer', {})
        spot_data = all_optimization_data.get('spot', {})
        ri_sp_data = all_optimization_data.get('ri_sp', {})

        # Calculate total savings potential
        total_savings = (
            cw_data.get('savings', 0) +
            ebs_data.get('total_savings', 0) +
            compute_opt_data.get('total_savings', 0) +
            spot_data.get('savings', 0) +
            ri_sp_data.get('potential_savings', 0)
        )

        print(f"\n💰 TOTAL OPTIMIZATION POTENTIAL: ${total_savings:,.2f}/month")
        print(f"💰 ANNUAL SAVINGS POTENTIAL: ${total_savings * 12:,.2f}\n")

        # Group actions by priority
        immediate_actions = []
        high_priority_actions = []
        medium_priority_actions = []

        # Immediate: EBS orphaned volumes
        if ebs_data.get('unattached_count', 0) > 0:
            immediate_actions.append({
                'title': f"Delete {ebs_data['unattached_count']} orphaned EBS volumes",
                'savings': ebs_data.get('unattached_savings', 0),
                'effort': '< 1 hour',
                'command': 'Use AWS Console or: aws ec2 delete-volume --volume-id <vol-id>'
            })

        # Immediate: Stop idle instances
        idle_count = len(cw_data.get('idle', []))
        if idle_count > 0:
            immediate_actions.append({
                'title': f"Stop or terminate {idle_count} idle instances",
                'savings': idle_count * 50,
                'effort': '< 1 hour',
                'command': 'aws ec2 stop-instances --instance-ids <instance-id>'
            })

        # High Priority: Savings Plans
        if ri_sp_data.get('potential_savings', 0) > 0:
            high_priority_actions.append({
                'title': 'Purchase recommended Savings Plans',
                'savings': ri_sp_data['potential_savings'],
                'effort': '< 1 day',
                'command': 'Review recommendations in AWS Cost Explorer → Savings Plans'
            })

        # High Priority: Compute Optimizer recommendations
        if compute_opt_data.get('total_savings', 0) > 0:
            high_priority_actions.append({
                'title': f"Implement {compute_opt_data.get('count', 0)} Compute Optimizer recommendations",
                'savings': compute_opt_data['total_savings'],
                'effort': '< 1 week',
                'command': 'Modify instance types using AWS Console or CLI'
            })

        # Medium Priority: gp2 to gp3 migration
        if ebs_data.get('gp2_migration_savings', 0) > 0:
            medium_priority_actions.append({
                'title': 'Migrate gp2 volumes to gp3',
                'savings': ebs_data['gp2_migration_savings'],
                'effort': '< 1 week',
                'command': 'aws ec2 modify-volume --volume-id <vol-id> --volume-type gp3'
            })

        # Medium Priority: Spot migration
        if spot_data.get('savings', 0) > 0:
            medium_priority_actions.append({
                'title': f"Migrate {spot_data.get('eligible_count', 0)} instances to Spot",
                'savings': spot_data['savings'],
                'effort': '< 1 month',
                'command': 'Use EC2 Spot Instances or Auto Scaling with mixed instances policy'
            })

        # Print actions
        if immediate_actions:
            print("🚨 IMMEDIATE ACTIONS (< 1 hour):\n")
            for i, action in enumerate(immediate_actions, 1):
                print(f"{i}. {action['title']}")
                print(f"   💰 Savings: ${action['savings']:,.2f}/month")
                print(f"   ⏱️  Effort: {action['effort']}")
                print(f"   📝 Command: {action['command']}\n")

        if high_priority_actions:
            print("⭐ HIGH PRIORITY (< 1 week):\n")
            for i, action in enumerate(high_priority_actions, 1):
                print(f"{i}. {action['title']}")
                print(f"   💰 Savings: ${action['savings']:,.2f}/month")
                print(f"   ⏱️  Effort: {action['effort']}")
                print(f"   📝 Action: {action['command']}\n")

        if medium_priority_actions:
            print("📋 MEDIUM PRIORITY (< 1 month):\n")
            for i, action in enumerate(medium_priority_actions, 1):
                print(f"{i}. {action['title']}")
                print(f"   💰 Savings: ${action['savings']:,.2f}/month")
                print(f"   ⏱️  Effort: {action['effort']}")
                print(f"   📝 Action: {action['command']}\n")

        return {
            'total_savings': total_savings,
            'immediate': immediate_actions,
            'high_priority': high_priority_actions,
            'medium_priority': medium_priority_actions
        }

    def save_detailed_report(self, all_data):
        """Save detailed JSON report"""
        filename = f"ec2_cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(all_data, f, indent=2, cls=DecimalEncoder)

        print(f"\n📄 Detailed report saved to: {filename}")

    def run_analysis(self, save_report: bool = True) -> dict:
        """
        Run complete EC2 cost analysis.

        Args:
            save_report: If True, save JSON report to file. Default True.

        Returns:
            Dictionary containing all analysis data.
        """
        account_display = f" - {self.account_name}" if self.account_name else ""
        print("\n" + "="*80)
        print(f"EC2 COST ANALYSIS - LAST 3 MONTHS{account_display}")
        print("="*80)
        print()

        # =================================================================
        # PHASE 1: COST EXPLORER DATA (Existing)
        # =================================================================
        time_data = self.fetch_ec2_costs_by_time()
        instance_type_data = self.fetch_costs_by_instance_type()
        region_data = self.fetch_costs_by_region()
        usage_type_data = self.fetch_costs_by_usage_type()

        # Fetch instance type costs broken down by Component tag (for stacked bar charts)
        instance_type_by_tag_data = self.fetch_costs_by_instance_type_and_tag('Component')
        instance_type_by_component = self.analyze_instance_type_by_tag(instance_type_by_tag_data, 'Component')

        # Fetch total AWS costs (all services)
        total_aws_time_data = self.fetch_total_aws_costs_by_time()
        service_breakdown_data = self.fetch_costs_by_service()

        # Analyze Cost Explorer data
        monthly_costs = self.analyze_time_trends(time_data)
        instance_costs = self.analyze_instance_types(instance_type_data)
        region_costs = self.analyze_regions(region_data)
        usage_costs = self.analyze_usage_types(usage_type_data)

        # Analyze total AWS costs
        total_aws_monthly_costs = self.analyze_total_aws_costs(total_aws_time_data)
        service_costs = self.analyze_service_costs(service_breakdown_data)

        # Fetch and analyze tags from Cost Explorer
        tag_keys = self.fetch_available_tag_keys()
        tag_analysis = {}

        if tag_keys:
            print(f"\n✓ Found {len(tag_keys)} tag keys in use")

            # Common important tags to prioritize
            priority_tags = [
                'Component', 'component',  # Important for instance type breakdown
                'Name', 'name',
                'Environment', 'environment', 'env',
                'Project', 'project',
                'Application', 'application', 'app',
                'Team', 'team',
                'Owner', 'owner',
                'CostCenter', 'cost-center', 'costcenter',
                'Department', 'department'
            ]

            # Analyze priority tags first
            analyzed_tags = set()
            for priority_tag in priority_tags:
                if priority_tag in tag_keys and priority_tag not in analyzed_tags:
                    tag_data = self.fetch_costs_by_tag(priority_tag)
                    tag_analysis[priority_tag] = self.analyze_tags(priority_tag, tag_data)
                    analyzed_tags.add(priority_tag)

            # Analyze remaining tags (up to 5 more)
            remaining_tags = [tag for tag in tag_keys if tag not in analyzed_tags]
            for tag_key in remaining_tags[:5]:
                tag_data = self.fetch_costs_by_tag(tag_key)
                tag_analysis[tag_key] = self.analyze_tags(tag_key, tag_data)
                analyzed_tags.add(tag_key)

            if len(tag_keys) > len(analyzed_tags):
                print(f"\n💡 {len(tag_keys) - len(analyzed_tags)} additional tag keys found but not analyzed")
                print(f"   Other tags: {', '.join([t for t in tag_keys if t not in analyzed_tags][:10])}")

        else:
            print("\n⚠️  No tags found - resources may not be tagged for cost allocation")

        # =================================================================
        # PHASE 2: ENHANCED OPTIMIZATION DATA (New)
        # =================================================================
        print("\n" + "="*80)
        print("FETCHING ENHANCED OPTIMIZATION DATA")
        print("="*80)
        print()

        # Fetch new data sources
        cw_data = self.fetch_cloudwatch_metrics()
        ebs_data = self.fetch_ebs_volumes()
        compute_opt_recommendations = self.fetch_compute_optimizer_recommendations()

        # Get instance types from cost data for spot pricing
        top_instance_types = list(instance_costs.keys())
        spot_prices = self.fetch_spot_pricing(top_instance_types)

        tag_compliance_data = self.fetch_tag_compliance()
        ri_sp_data = self.fetch_ri_savings_plans_data()

        # Fiscal Year Forecast (April to April - all AWS spend)
        fiscal_year_forecast_data = self.fetch_fiscal_year_forecast()

        # =================================================================
        # PHASE 3: ENHANCED ANALYSIS (New - Prioritized)
        # =================================================================

        # 1. RI/Savings Plans (HIGH PRIORITY - per user goal)
        ri_sp_analysis = self.analyze_ri_savings_plans(ri_sp_data)

        # 2. Compute Optimizer (HIGH PRIORITY)
        compute_opt_analysis = self.analyze_compute_optimizer(compute_opt_recommendations)

        # 3. EBS Optimization (QUICK WINS)
        ebs_analysis = self.analyze_ebs_optimization(ebs_data)

        # 4. CloudWatch Utilization
        cw_analysis = self.analyze_cloudwatch_utilization(cw_data, instance_costs)

        # 5. Spot Opportunities
        spot_analysis = self.analyze_spot_opportunities(spot_prices, instance_costs, cw_data)

        # 6. Tag Compliance (Detailed)
        tag_compliance_analysis = self.analyze_tag_compliance_detailed(tag_compliance_data)

        # 7. Fiscal Year Forecast (April to April)
        fiscal_year_forecast = self.analyze_fiscal_year_forecast(fiscal_year_forecast_data)

        # =================================================================
        # PHASE 4: GENERATE ACTION PLAN
        # =================================================================
        all_optimization_data = {
            'cloudwatch': cw_analysis,
            'ebs': ebs_analysis,
            'compute_optimizer': compute_opt_analysis,
            'spot': spot_analysis,
            'ri_sp': ri_sp_analysis
        }

        action_plan = self.generate_action_plan(all_optimization_data)

        # Generate legacy insights (for backwards compatibility)
        self.generate_insights(monthly_costs, instance_costs, region_costs, usage_costs)

        # =================================================================
        # PHASE 5: SAVE COMPREHENSIVE REPORT
        # =================================================================
        all_data = {
            'generated_at': datetime.now().isoformat(),
            'account_id': self.account_id,
            'account_name': self.account_name,
            'time_period': self.get_date_range(),

            # Cost Explorer data - EC2 specific
            'monthly_costs': monthly_costs,
            'instance_type_costs': instance_costs,
            'instance_type_by_component': instance_type_by_component,  # For stacked bar charts
            'region_costs': region_costs,
            'usage_type_costs': usage_costs,
            'tag_keys': tag_keys,
            'tag_analysis': tag_analysis,

            # Total AWS costs (all services)
            'total_aws_monthly_costs': total_aws_monthly_costs,
            'service_costs': service_costs,

            # Enhanced optimization data
            'cloudwatch_analysis': cw_analysis,
            'ebs_analysis': ebs_analysis,
            'compute_optimizer_analysis': compute_opt_analysis,
            'spot_analysis': spot_analysis,
            'ri_sp_analysis': ri_sp_analysis,
            'tag_compliance_analysis': tag_compliance_analysis,

            # Fiscal Year Forecast (April to April - all AWS spend)
            'fiscal_year_forecast': fiscal_year_forecast,

            # Action plan
            'action_plan': action_plan
        }

        if save_report:
            self.save_detailed_report(all_data)

        print("\n✓ Analysis complete!")
        return all_data


if __name__ == '__main__':
    analyzer = EC2CostAnalyzer()
    analyzer.run_analysis()
