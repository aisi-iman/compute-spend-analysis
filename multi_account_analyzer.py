#!/usr/bin/env python3
"""
Multi-Account EC2 Cost Analyzer

Runs cost analysis across multiple AWS accounts and generates a combined report.
Uses AWS CLI profiles configured via SSO for authentication.

Usage:
    uv run python multi_account_analyzer.py
    uv run python multi_account_analyzer.py -c custom_config.yaml
"""

import boto3
import json
import sys
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ec2_cost_analyzer import EC2CostAnalyzer, DecimalEncoder


class AccountConfig:
    """Configuration for a single AWS account."""

    def __init__(self, profile: str, name: str, account_id: str = "", enabled: bool = True):
        self.profile = profile
        self.name = name
        self.account_id = account_id
        self.enabled = enabled


class MultiAccountConfig:
    """Configuration for multi-account analysis."""

    def __init__(
        self,
        accounts: List[AccountConfig],
        continue_on_error: bool = True,
        include_zero_spend: bool = False
    ):
        self.accounts = accounts
        self.continue_on_error = continue_on_error
        self.include_zero_spend = include_zero_spend


def load_config(config_path: str = "accounts_config.yaml") -> MultiAccountConfig:
    """
    Load accounts configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        MultiAccountConfig object
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Run 'uv run python setup_multi_account.py' to generate it."
        )

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    accounts = []
    for acc in data.get('accounts', []):
        accounts.append(AccountConfig(
            profile=acc['profile'],
            name=acc.get('name', acc['profile']),
            account_id=acc.get('account_id', ''),
            enabled=acc.get('enabled', True)
        ))

    settings = data.get('settings', {})

    return MultiAccountConfig(
        accounts=accounts,
        continue_on_error=settings.get('continue_on_error', True),
        include_zero_spend=settings.get('include_zero_spend_accounts', False)
    )


class MultiAccountAnalyzer:
    """Orchestrates cost analysis across multiple AWS accounts."""

    def __init__(self, config: MultiAccountConfig):
        self.config = config
        self.results: Dict[str, Dict] = {}
        self.errors: Dict[str, str] = {}

    def get_session_for_profile(self, profile_name: str) -> boto3.Session:
        """
        Create a boto3 Session using the specified AWS CLI profile.

        Args:
            profile_name: AWS CLI profile name

        Returns:
            boto3.Session configured with the profile
        """
        try:
            session = boto3.Session(profile_name=profile_name)
            # Test the session
            sts = session.client('sts')
            sts.get_caller_identity()
            return session
        except Exception as e:
            raise RuntimeError(
                f"Failed to create session for profile '{profile_name}': {e}\n"
                f"Try running: aws sso login --profile {profile_name}"
            )

    def analyze_account(self, account: AccountConfig) -> Dict[str, Any]:
        """
        Run analysis for a single account.

        Args:
            account: Account configuration

        Returns:
            Analysis result dictionary
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING: {account.name} (profile: {account.profile})")
        print(f"{'='*60}")

        session = self.get_session_for_profile(account.profile)
        analyzer = EC2CostAnalyzer(session=session, account_name=account.name)

        # Run analysis without saving individual report
        result = analyzer.run_analysis(save_report=False)
        return result

    def run_all_accounts(self) -> Dict[str, Any]:
        """
        Run analysis across all configured accounts.

        Returns:
            Combined report dictionary
        """
        enabled_accounts = [a for a in self.config.accounts if a.enabled]

        if not enabled_accounts:
            raise ValueError("No enabled accounts found in configuration")

        print("\n" + "=" * 80)
        print("MULTI-ACCOUNT EC2 COST ANALYSIS")
        print("=" * 80)
        print(f"\nAnalyzing {len(enabled_accounts)} accounts...")

        account_reports = []

        for account in enabled_accounts:
            try:
                result = self.analyze_account(account)
                account_reports.append(result)
                self.results[result['account_id']] = result
            except Exception as e:
                error_msg = str(e)
                self.errors[account.profile] = error_msg
                print(f"\nERROR analyzing {account.name}: {error_msg}")

                if not self.config.continue_on_error:
                    raise

        if not account_reports:
            raise RuntimeError("No accounts were successfully analyzed")

        # Generate combined report
        combined = self.aggregate_reports(account_reports)
        return combined

    def aggregate_reports(self, reports: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate individual account reports into a combined report.

        Args:
            reports: List of individual account analysis results

        Returns:
            Combined report dictionary
        """
        print(f"\n{'='*60}")
        print("AGGREGATING REPORTS")
        print(f"{'='*60}")

        # Per-account summaries
        account_reports = {}
        for r in reports:
            account_id = r['account_id']
            total_cost = sum(m['cost'] for m in r.get('monthly_costs', []))

            account_reports[account_id] = {
                'account_name': r.get('account_name', 'Unknown'),
                'account_id': account_id,
                'total_cost': total_cost,
                'monthly_costs': r.get('monthly_costs', []),
                'instance_type_costs': r.get('instance_type_costs', {}),
                'instance_type_by_component': r.get('instance_type_by_component', {}),
                'region_costs': r.get('region_costs', {}),
                'tag_keys': r.get('tag_keys', []),
                'tag_analysis': r.get('tag_analysis', {}),
                'cloudwatch_analysis': r.get('cloudwatch_analysis', {}),
                'ebs_analysis': r.get('ebs_analysis', {}),
                'spot_analysis': r.get('spot_analysis', {}),
                'ri_sp_analysis': r.get('ri_sp_analysis', {}),
                'action_plan': r.get('action_plan', {}),
                'fiscal_year_forecast': r.get('fiscal_year_forecast', {})
            }

        # Calculate aggregated totals
        aggregated = self._aggregate_totals(reports)

        combined = {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'multi_account',
            'accounts_analyzed': len(reports),
            'accounts_failed': len(self.errors),
            'errors': self.errors,
            'account_reports': account_reports,
            'aggregated': aggregated
        }

        return combined

    def _aggregate_totals(self, reports: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregated totals across all accounts."""

        # Aggregate EC2 monthly costs by period
        monthly_by_period = defaultdict(lambda: {'cost': 0, 'usage': 0})
        for report in reports:
            for month in report.get('monthly_costs', []):
                period = month['period']
                monthly_by_period[period]['cost'] += month['cost']
                monthly_by_period[period]['usage'] += month.get('usage', 0)

        aggregated_monthly = [
            {'period': period, **values}
            for period, values in sorted(monthly_by_period.items())
        ]

        # Aggregate total AWS monthly costs by period
        total_aws_by_period = defaultdict(lambda: {'cost': 0})
        for report in reports:
            for month in report.get('total_aws_monthly_costs', []):
                period = month['period']
                total_aws_by_period[period]['cost'] += month['cost']

        aggregated_total_aws_monthly = [
            {'period': period, **values}
            for period, values in sorted(total_aws_by_period.items())
        ]

        # Aggregate service costs
        service_costs = defaultdict(float)
        for report in reports:
            for service, cost in report.get('service_costs', {}).items():
                service_costs[service] += cost

        # Aggregate instance type costs
        instance_costs = defaultdict(float)
        for report in reports:
            for instance_type, cost in report.get('instance_type_costs', {}).items():
                instance_costs[instance_type] += cost

        # Aggregate region costs
        region_costs = defaultdict(float)
        for report in reports:
            for region, cost in report.get('region_costs', {}).items():
                region_costs[region] += cost

        # Aggregate savings potential from action plans
        total_savings = 0
        for report in reports:
            action_plan = report.get('action_plan', {})
            total_savings += action_plan.get('total_savings', 0)

        # Aggregate fiscal year forecast
        fy_totals = {
            'total_actual': 0,
            'total_forecast': 0,
            'total_projected': 0
        }
        for report in reports:
            fy = report.get('fiscal_year_forecast', {})
            fy_totals['total_actual'] += fy.get('total_actual', 0)
            fy_totals['total_forecast'] += fy.get('total_forecast', 0)
            fy_totals['total_projected'] += fy.get('total_projected', 0)

        # Calculate per-account cost breakdown (EC2)
        account_ec2_costs = []
        for report in reports:
            total = sum(m['cost'] for m in report.get('monthly_costs', []))
            account_ec2_costs.append({
                'account_id': report['account_id'],
                'account_name': report.get('account_name', 'Unknown'),
                'total_cost': total
            })
        account_ec2_costs.sort(key=lambda x: x['total_cost'], reverse=True)

        # Calculate per-account cost breakdown (Total AWS)
        account_total_costs = []
        for report in reports:
            total = sum(m['cost'] for m in report.get('total_aws_monthly_costs', []))
            account_total_costs.append({
                'account_id': report['account_id'],
                'account_name': report.get('account_name', 'Unknown'),
                'total_cost': total
            })
        account_total_costs.sort(key=lambda x: x['total_cost'], reverse=True)

        # Aggregate tag analysis across all accounts
        all_tag_keys = set()
        aggregated_tag_analysis = {}
        for report in reports:
            # Collect all unique tag keys
            tag_keys = report.get('tag_keys', [])
            all_tag_keys.update(tag_keys)

            # Aggregate tag costs
            tag_analysis = report.get('tag_analysis', {})
            for tag_key, tag_data in tag_analysis.items():
                if tag_key not in aggregated_tag_analysis:
                    aggregated_tag_analysis[tag_key] = defaultdict(float)
                # tag_data is a dict of {tag_value: cost}
                if isinstance(tag_data, dict):
                    for tag_value, cost in tag_data.items():
                        if isinstance(cost, (int, float)):
                            aggregated_tag_analysis[tag_key][tag_value] += cost

        # Convert defaultdicts to regular dicts and sort by cost
        for tag_key in aggregated_tag_analysis:
            aggregated_tag_analysis[tag_key] = dict(sorted(
                aggregated_tag_analysis[tag_key].items(),
                key=lambda x: x[1], reverse=True
            ))

        # Aggregate instance_type_by_component across all accounts
        aggregated_instance_by_component = defaultdict(lambda: defaultdict(float))
        for report in reports:
            instance_by_component = report.get('instance_type_by_component', {})
            for instance_type, data in instance_by_component.items():
                by_tag = data.get('by_tag', {}) if isinstance(data, dict) else {}
                for tag_value, cost in by_tag.items():
                    aggregated_instance_by_component[instance_type][tag_value] += cost

        # Convert to final format with totals
        instance_type_by_component_final = {}
        for instance_type, tag_costs in aggregated_instance_by_component.items():
            instance_type_by_component_final[instance_type] = {
                'by_tag': dict(tag_costs),
                'total': sum(tag_costs.values())
            }
        # Sort by total cost descending
        instance_type_by_component_final = dict(sorted(
            instance_type_by_component_final.items(),
            key=lambda x: x[1]['total'], reverse=True
        ))

        return {
            # EC2 costs
            'total_ec2_cost': sum(m['cost'] for m in aggregated_monthly),
            'monthly_costs': aggregated_monthly,
            'instance_type_costs': dict(sorted(
                instance_costs.items(), key=lambda x: x[1], reverse=True
            )),
            'instance_type_by_component': instance_type_by_component_final,
            'region_costs': dict(sorted(
                region_costs.items(), key=lambda x: x[1], reverse=True
            )),
            'ec2_cost_by_account': account_ec2_costs,

            # Total AWS costs (all services)
            'total_aws_cost': sum(m['cost'] for m in aggregated_total_aws_monthly),
            'total_aws_monthly_costs': aggregated_total_aws_monthly,
            'service_costs': dict(sorted(
                service_costs.items(), key=lambda x: x[1], reverse=True
            )),
            'total_aws_cost_by_account': account_total_costs,

            # Tags
            'tag_keys': sorted(list(all_tag_keys)),
            'tag_analysis': aggregated_tag_analysis,

            # Other
            'total_optimization_potential': total_savings,
            'fiscal_year_forecast': fy_totals,
            # Keep backwards compatibility
            'total_cost': sum(m['cost'] for m in aggregated_monthly),
            'cost_by_account': account_ec2_costs
        }

    def save_combined_report(self, report: Dict, output_dir: str = ".") -> str:
        """
        Save the combined report to a JSON file.

        Args:
            report: Combined report dictionary
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"multi_account_cost_report_{timestamp}.json"
        filepath = Path(output_dir) / filename

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, cls=DecimalEncoder)

        print(f"\n📄 Combined report saved to: {filepath}")
        return str(filepath)

    def print_summary(self, report: Dict):
        """Print a summary of the combined report."""
        agg = report['aggregated']

        print(f"\n{'='*80}")
        print("MULTI-ACCOUNT ANALYSIS SUMMARY")
        print(f"{'='*80}")

        print(f"\nAccounts analyzed: {report['accounts_analyzed']}")
        print(f"Accounts failed: {report['accounts_failed']}")

        # Total AWS costs (all services)
        total_aws = agg.get('total_aws_cost', 0)
        total_ec2 = agg.get('total_ec2_cost', agg.get('total_cost', 0))
        ec2_pct = (total_ec2 / total_aws * 100) if total_aws > 0 else 0

        print(f"\n📊 COST SUMMARY (3-month)")
        print(f"   Total AWS Cost (all services): ${total_aws:>15,.2f}")
        print(f"   EC2 Cost:                      ${total_ec2:>15,.2f} ({ec2_pct:.1f}% of total)")
        print(f"💰 Total optimization potential:  ${agg['total_optimization_potential']:,.2f}/month")

        # Top services
        if agg.get('service_costs'):
            print(f"\n🔧 Top AWS Services:")
            for service, cost in list(agg['service_costs'].items())[:5]:
                pct = (cost / total_aws * 100) if total_aws > 0 else 0
                service_short = service[:35] + "..." if len(service) > 35 else service
                print(f"   {service_short:40s} ${cost:>12,.2f} ({pct:5.1f}%)")

        # Cost by account (total AWS)
        print(f"\n📋 Total AWS Cost by Account:")
        account_costs = agg.get('total_aws_cost_by_account', agg.get('cost_by_account', []))
        for acc in account_costs[:10]:
            pct = (acc['total_cost'] / total_aws * 100) if total_aws > 0 else 0
            print(f"   {acc['account_name']:30s} ${acc['total_cost']:>12,.2f} ({pct:5.1f}%)")

        if len(account_costs) > 10:
            print(f"   ... and {len(account_costs) - 10} more accounts")

        print(f"\n📅 Fiscal Year Forecast (All Accounts - All Services):")
        fy = agg['fiscal_year_forecast']
        print(f"   Actual (YTD):      ${fy['total_actual']:>15,.2f}")
        print(f"   Forecasted:        ${fy['total_forecast']:>15,.2f}")
        print(f"   Projected Total:   ${fy['total_projected']:>15,.2f}")

        print(f"\n🌍 Top Regions (EC2):")
        for region, cost in list(agg['region_costs'].items())[:5]:
            pct = (cost / total_ec2 * 100) if total_ec2 > 0 else 0
            print(f"   {region:20s} ${cost:>12,.2f} ({pct:5.1f}%)")

        if report['errors']:
            print(f"\n⚠️  Errors:")
            for profile, error in report['errors'].items():
                print(f"   {profile}: {error[:80]}...")


def main():
    """Main entry point for multi-account analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run EC2 cost analysis across multiple AWS accounts'
    )
    parser.add_argument(
        '-c', '--config',
        default='accounts_config.yaml',
        help='Path to accounts configuration file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='.',
        help='Output directory for the combined report'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        print(f"Found {len(config.accounts)} accounts in configuration")

        # Run analysis
        analyzer = MultiAccountAnalyzer(config)
        report = analyzer.run_all_accounts()

        # Save report
        filepath = analyzer.save_combined_report(report, args.output_dir)

        # Print summary
        analyzer.print_summary(report)

        print(f"\n{'='*80}")
        print("Analysis complete!")
        print(f"{'='*80}")
        print(f"\nReport saved to: {filepath}")
        print("\nNext steps:")
        print("  1. Open the Jupyter notebook to view the report")
        print("  2. Update REPORT_JSON_PATH to the new file")
        print("  3. Run all cells to see multi-account visualizations")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo set up multi-account analysis:")
        print("  1. Run: uv run python setup_multi_account.py")
        print("  2. Follow the prompts to discover your accounts")
        print("  3. Run: uv run python multi_account_analyzer.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
