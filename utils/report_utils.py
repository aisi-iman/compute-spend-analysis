"""
EC2 Cost Optimization Report - Utility Functions
Helper functions for loading data, creating visualizations, and generating commands.
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime


# =====================================================================
# DATA LOADING FUNCTIONS
# =====================================================================

def load_cost_report(json_path: str) -> Dict[str, Any]:
    """
    Load EC2 cost report JSON file and validate structure.

    Args:
        json_path: Path to the JSON report file

    Returns:
        Dictionary containing the cost report data

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        ValueError: If required fields are missing
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Validate required fields
        required_fields = ['generated_at', 'time_period', 'monthly_costs',
                          'instance_type_costs', 'region_costs']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"Cost report file not found: {json_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in report file: {e}", e.doc, e.pos)


def parse_monthly_costs(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert monthly costs to DataFrame."""
    monthly_data = data.get('monthly_costs', [])
    df = pd.DataFrame(monthly_data)
    if not df.empty:
        df['period_label'] = df['period'].apply(lambda x: x.split(' to ')[0][:7])  # YYYY-MM format
    return df


def parse_instance_type_costs(data: Dict[str, Any], top_n: int = 15) -> pd.DataFrame:
    """Convert instance type costs to DataFrame and get top N."""
    costs = data.get('instance_type_costs', {})
    df = pd.DataFrame(list(costs.items()), columns=['instance_type', 'cost'])
    df = df.sort_values('cost', ascending=False).head(top_n)
    total_cost = sum(costs.values())
    df['percentage'] = (df['cost'] / total_cost * 100).round(1)
    return df


def parse_region_costs(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert region costs to DataFrame."""
    costs = data.get('region_costs', {})
    df = pd.DataFrame(list(costs.items()), columns=['region', 'cost'])
    df = df.sort_values('cost', ascending=False)
    total_cost = df['cost'].sum()
    df['percentage'] = (df['cost'] / total_cost * 100).round(1)
    return df


def parse_tag_costs(data: Dict[str, Any], tag_key: str, top_n: int = 20) -> pd.DataFrame:
    """Convert tag-based costs to DataFrame."""
    tag_analysis = data.get('tag_analysis', {})
    tag_costs = tag_analysis.get(tag_key, {})

    df = pd.DataFrame(list(tag_costs.items()), columns=['tag_value', 'cost'])
    df = df.sort_values('cost', ascending=False).head(top_n)
    total_cost = sum(tag_costs.values())
    df['percentage'] = (df['cost'] / total_cost * 100).round(1)

    # Clean up tag values (remove "TagKey$" prefix)
    df['tag_value'] = df['tag_value'].apply(lambda x: x.split('$', 1)[-1] if '$' in x else x)
    df['tag_value'] = df['tag_value'].apply(lambda x: '(untagged)' if x == '' else x)

    return df


def parse_cloudwatch_instances(data: Dict[str, Any], category: str) -> pd.DataFrame:
    """
    Parse CloudWatch instance data for a specific category.

    Args:
        data: Cost report data
        category: 'idle', 'underutilized', or 'high_utilization'

    Returns:
        DataFrame with instance details
    """
    cw_analysis = data.get('cloudwatch_analysis', {})
    instances = cw_analysis.get(category, [])

    if not instances:
        return pd.DataFrame()

    df = pd.DataFrame(instances)

    # Extract relevant fields
    if 'Tags' in df.columns:
        df['Name'] = df['Tags'].apply(lambda x: x.get('Name', 'N/A') if isinstance(x, dict) else 'N/A')
        df['owner'] = df['Tags'].apply(lambda x: x.get('owner', 'N/A') if isinstance(x, dict) else 'N/A')
        df['Project'] = df['Tags'].apply(lambda x: x.get('Project', 'N/A') if isinstance(x, dict) else 'N/A')

    # Rename columns for clarity
    if 'InstanceId' in df.columns:
        df = df.rename(columns={'InstanceId': 'instance_id'})
    if 'InstanceType' in df.columns:
        df = df.rename(columns={'InstanceType': 'instance_type'})
    if 'AvgCPU' in df.columns:
        df = df.rename(columns={'AvgCPU': 'avg_cpu'})
    if 'MaxCPU' in df.columns:
        df = df.rename(columns={'MaxCPU': 'max_cpu'})

    # Add instance_type as 'N/A' if it doesn't exist (not included in CloudWatch data)
    if 'instance_type' not in df.columns:
        df['instance_type'] = 'N/A'

    return df


def parse_ebs_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse EBS storage optimization data."""
    ebs = data.get('ebs_analysis', {})
    return {
        'unattached_count': ebs.get('unattached_count', 0),
        'unattached_savings': ebs.get('unattached_savings', 0),
        'gp2_migration_savings': ebs.get('gp2_migration_savings', 0),
        'old_snapshots': ebs.get('old_snapshots', 0),
        'total_savings': ebs.get('total_savings', 0),
        'unattached_volumes': ebs.get('unattached_volumes', [])
    }


def parse_spot_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse spot instance opportunity data."""
    spot = data.get('spot_analysis', {})
    return {
        'eligible_count': spot.get('eligible_count', 0),
        'savings': spot.get('savings', 0),
        'eligible_instances': spot.get('eligible_instances', [])
    }


def parse_ri_sp_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse RI/Savings Plans data."""
    ri_sp = data.get('ri_sp_analysis', {})
    return {
        'coverage': ri_sp.get('coverage', 0),
        'utilization': ri_sp.get('utilization', 0),
        'potential_savings': ri_sp.get('potential_savings', 0),
        'recommendations': ri_sp.get('recommendations', [])
    }


def parse_total_aws_costs(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert total AWS monthly costs to DataFrame."""
    monthly_data = data.get('total_aws_monthly_costs', [])
    df = pd.DataFrame(monthly_data)
    if not df.empty and 'period' in df.columns:
        df['period_label'] = df['period'].apply(lambda x: x.split(' to ')[0][:7])
    return df


def parse_service_costs(data: Dict[str, Any], top_n: int = 15) -> pd.DataFrame:
    """Convert service costs to DataFrame and get top N."""
    costs = data.get('service_costs', {})
    if not costs:
        return pd.DataFrame()

    df = pd.DataFrame(list(costs.items()), columns=['service', 'cost'])
    df = df.sort_values('cost', ascending=False).head(top_n)
    total_cost = sum(costs.values())
    df['percentage'] = (df['cost'] / total_cost * 100).round(1)
    return df


def get_ec2_vs_other_costs(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate EC2 costs vs other AWS services.

    Returns:
        Dictionary with ec2_cost, other_cost, total_cost, ec2_percentage
    """
    # Get total AWS cost
    total_aws_monthly = data.get('total_aws_monthly_costs', [])
    total_aws_cost = sum(m.get('cost', 0) for m in total_aws_monthly)

    # Get EC2 cost
    ec2_monthly = data.get('monthly_costs', [])
    ec2_cost = sum(m.get('cost', 0) for m in ec2_monthly)

    # Calculate other
    other_cost = total_aws_cost - ec2_cost

    ec2_pct = (ec2_cost / total_aws_cost * 100) if total_aws_cost > 0 else 0

    return {
        'ec2_cost': ec2_cost,
        'other_cost': other_cost,
        'total_cost': total_aws_cost,
        'ec2_percentage': ec2_pct
    }


# =====================================================================
# MULTI-ACCOUNT DATA FUNCTIONS
# =====================================================================

def load_multi_account_report(json_path: str) -> Dict[str, Any]:
    """
    Load multi-account cost report JSON file and validate structure.

    Args:
        json_path: Path to the JSON report file

    Returns:
        Dictionary containing the multi-account cost report data

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        ValueError: If required fields are missing or not a multi-account report
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Validate it's a multi-account report
        if data.get('report_type') != 'multi_account':
            raise ValueError("Not a multi-account report. Expected 'report_type': 'multi_account'")

        # Validate required fields
        required_fields = ['generated_at', 'accounts_analyzed', 'account_reports', 'aggregated']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"Cost report file not found: {json_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in report file: {e}", e.doc, e.pos)


def get_account_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get list of accounts from multi-account report.

    Args:
        data: Multi-account cost report data

    Returns:
        List of dicts with account_id, account_name, total_cost, sorted by cost descending
    """
    account_reports = data.get('account_reports', {})
    accounts = []

    for account_id, account_data in account_reports.items():
        accounts.append({
            'account_id': account_id,
            'account_name': account_data.get('account_name', account_id),
            'total_cost': account_data.get('total_cost', 0)
        })

    # Sort by cost descending
    accounts.sort(key=lambda x: x['total_cost'], reverse=True)
    return accounts


def get_account_data(data: Dict[str, Any], account_id: str) -> Dict[str, Any]:
    """
    Extract single account's data from multi-account report.

    Args:
        data: Multi-account cost report data
        account_id: The account ID to extract

    Returns:
        Dictionary containing the account's cost data
    """
    account_reports = data.get('account_reports', {})
    return account_reports.get(account_id, {})


def get_aggregated_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get aggregated data section from multi-account report.

    Args:
        data: Multi-account cost report data

    Returns:
        Dictionary containing aggregated cost data across all accounts
    """
    return data.get('aggregated', {})


def parse_account_costs(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse account costs into DataFrame for visualization.

    Args:
        data: Multi-account cost report data

    Returns:
        DataFrame with columns: account_name, account_id, total_cost, percentage
    """
    aggregated = data.get('aggregated', {})
    account_costs = aggregated.get('ec2_cost_by_account', [])

    if not account_costs:
        return pd.DataFrame()

    df = pd.DataFrame(account_costs)

    # Ensure required columns exist
    if 'account_name' not in df.columns:
        df['account_name'] = df.get('account_id', 'Unknown')

    # Calculate percentage
    total = df['total_cost'].sum()
    df['percentage'] = (df['total_cost'] / total * 100).round(1) if total > 0 else 0

    # Sort by cost descending
    df = df.sort_values('total_cost', ascending=False)

    return df


def parse_fiscal_year_forecast(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse fiscal year forecast data (April to April).

    Args:
        data: Cost report data

    Returns:
        Dictionary containing:
        - fiscal_year: Date range string
        - fiscal_year_label: FY label (e.g., "FY2025/2026")
        - monthly_df: DataFrame with monthly breakdown
        - total_actual: Total actual spend to date
        - total_forecast: Total forecasted spend
        - total_projected: Total projected spend for the year
    """
    forecast = data.get('fiscal_year_forecast', {})

    result = {
        'fiscal_year': forecast.get('fiscal_year', 'N/A'),
        'fiscal_year_label': forecast.get('fiscal_year_label', 'N/A'),
        'total_actual': forecast.get('total_actual', 0),
        'total_forecast': forecast.get('total_forecast', 0),
        'total_projected': forecast.get('total_projected', 0),
        'monthly_df': pd.DataFrame()
    }

    monthly_data = forecast.get('monthly_data', [])
    if monthly_data:
        df = pd.DataFrame(monthly_data)
        # Ensure proper column order and formatting
        if 'month_label' in df.columns:
            df['month'] = df['month_label']
        if 'cost' in df.columns:
            df['amount'] = df['cost']
        result['monthly_df'] = df

    return result


# =====================================================================
# VISUALIZATION FUNCTIONS
# =====================================================================

def create_kpi_card(value: float, title: str, prefix: str = "$",
                    suffix: str = "", subtitle: str = "") -> go.Figure:
    """
    Create a KPI card visualization.

    Args:
        value: The numeric value to display
        title: Main title of the KPI
        prefix: Prefix for the value (e.g., "$")
        suffix: Suffix for the value (e.g., "/month")
        subtitle: Additional context below the value
    """
    formatted_value = f"{prefix}{value:,.2f}{suffix}"

    fig = go.Figure()

    fig.add_annotation(
        text=f"<b>{title}</b>",
        xref="paper", yref="paper",
        x=0.5, y=0.9,
        showarrow=False,
        font=dict(size=16, color="#666"),
    )

    fig.add_annotation(
        text=f"<b>{formatted_value}</b>",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=32, color="#1f77b4"),
    )

    if subtitle:
        fig.add_annotation(
            text=subtitle,
            xref="paper", yref="paper",
            x=0.5, y=0.2,
            showarrow=False,
            font=dict(size=12, color="#999"),
        )

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


def create_monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Create monthly cost trend line chart with dual axis for cost and usage."""
    fig = go.Figure()

    # Cost line
    fig.add_trace(go.Scatter(
        x=df['period_label'],
        y=df['cost'],
        name='Cost',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10),
        hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<extra></extra>',
        yaxis='y'
    ))

    # Usage line on secondary axis
    fig.add_trace(go.Scatter(
        x=df['period_label'],
        y=df['usage'],
        name='Usage (hours)',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Usage: %{y:,.0f} hours<extra></extra>',
        yaxis='y2'
    ))

    fig.update_layout(
        title='Monthly Cost Trend',
        xaxis=dict(title='Month'),
        yaxis=dict(title='Cost ($)', side='left', showgrid=True),
        yaxis2=dict(title='Usage (hours)', side='right', overlaying='y', showgrid=False),
        hovermode='x unified',
        height=400,
        legend=dict(x=0, y=1.1, orientation='h'),
        template='plotly_white'
    )

    return fig


def create_horizontal_bar_chart(df: pd.DataFrame, x_col: str, y_col: str,
                                 title: str, x_label: str = "Cost ($)") -> go.Figure:
    """Create horizontal bar chart for cost breakdowns."""
    fig = px.bar(
        df,
        y=y_col,
        x=x_col,
        orientation='h',
        title=title,
        labels={x_col: x_label, y_col: ''},
        text=df.apply(lambda row: f"${row[x_col]:,.0f} ({row['percentage']}%)"
                      if 'percentage' in df.columns else f"${row[x_col]:,.0f}", axis=1)
    )

    fig.update_traces(
        textposition='outside',
        marker_color='#1f77b4',
        hovertemplate='<b>%{y}</b><br>Cost: $%{x:,.2f}<extra></extra>'
    )

    fig.update_layout(
        height=max(400, len(df) * 30),
        xaxis_title=x_label,
        yaxis=dict(autorange='reversed'),
        template='plotly_white',
        showlegend=False
    )

    return fig


def create_pie_chart(df: pd.DataFrame, values_col: str, names_col: str,
                     title: str, hole: float = 0.3) -> go.Figure:
    """Create pie/donut chart for distributions."""
    fig = px.pie(
        df,
        values=values_col,
        names=names_col,
        title=title,
        hole=hole
    )

    fig.update_traces(
        textposition='inside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>'
    )

    fig.update_layout(
        height=400,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', yanchor='middle', y=0.5)
    )

    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                       hover_cols: List[str], title: str,
                       x_label: str, y_label: str) -> go.Figure:
    """Create scatter plot for correlation analysis."""
    hover_text = df[hover_cols].apply(
        lambda row: '<br>'.join([f"{col}: {row[col]}" for col in hover_cols]),
        axis=1
    )

    fig = go.Figure(data=go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        marker=dict(
            size=10,
            color=df[y_col] if y_col in df.columns else '#1f77b4',
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title=y_label)
        ),
        text=hover_text,
        hovertemplate='%{text}<br>' + f'{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.2f}}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        template='plotly_white'
    )

    return fig


def create_gauge_chart(value: float, title: str, max_value: float = 100,
                      threshold_good: float = 70, threshold_ok: float = 40) -> go.Figure:
    """Create gauge chart for coverage/utilization metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        delta={'reference': threshold_good},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold_ok], 'color': "lightgray"},
                {'range': [threshold_ok, threshold_good], 'color': "yellow"},
                {'range': [threshold_good, max_value], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_good
            }
        }
    ))

    fig.update_layout(height=300, template='plotly_white')

    return fig


def create_waterfall_chart(categories: List[str], values: List[float],
                           title: str = "Savings Potential") -> go.Figure:
    """Create waterfall chart for cumulative savings visualization."""
    fig = go.Figure(go.Waterfall(
        name="Savings",
        orientation="v",
        measure=["relative"] * (len(categories) - 1) + ["total"],
        x=categories,
        textposition="outside",
        text=[f"${v:,.0f}" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        height=400,
        template='plotly_white'
    )

    return fig


def create_ec2_vs_other_chart(data: Dict[str, Any]) -> go.Figure:
    """
    Create a donut chart showing EC2 costs vs other AWS services.

    Args:
        data: Cost report data containing monthly_costs and total_aws_monthly_costs

    Returns:
        Plotly Figure object
    """
    cost_breakdown = get_ec2_vs_other_costs(data)

    if cost_breakdown['total_cost'] == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No cost data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400, template='plotly_white')
        return fig

    labels = ['EC2 (Compute)', 'Other AWS Services']
    values = [cost_breakdown['ec2_cost'], cost_breakdown['other_cost']]
    colors = ['#1f77b4', '#ff7f0e']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(
            text=f'EC2 vs Other AWS Services (3-Month Total: ${cost_breakdown["total_cost"]:,.0f})',
            x=0.5,
            xanchor='center'
        ),
        height=400,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        annotations=[dict(
            text=f'EC2<br>{cost_breakdown["ec2_percentage"]:.1f}%',
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )]
    )

    return fig


def create_service_breakdown_chart(data: Dict[str, Any], top_n: int = 10) -> go.Figure:
    """
    Create a horizontal bar chart showing cost breakdown by AWS service.

    Args:
        data: Cost report data containing service_costs
        top_n: Number of top services to show

    Returns:
        Plotly Figure object
    """
    service_df = parse_service_costs(data, top_n=top_n)

    if service_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No service cost data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400, template='plotly_white')
        return fig

    # Truncate long service names
    service_df['service_short'] = service_df['service'].apply(
        lambda x: x[:40] + '...' if len(x) > 40 else x
    )

    fig = go.Figure(go.Bar(
        y=service_df['service_short'],
        x=service_df['cost'],
        orientation='h',
        text=service_df.apply(lambda row: f"${row['cost']:,.0f} ({row['percentage']}%)", axis=1),
        textposition='outside',
        marker_color=['#1f77b4' if 'EC2' in s or 'Compute' in s else '#ff7f0e'
                      for s in service_df['service']],
        hovertemplate='<b>%{y}</b><br>Cost: $%{x:,.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='AWS Cost Breakdown by Service (3-Month)',
        xaxis_title='Cost ($)',
        yaxis=dict(autorange='reversed'),
        height=max(400, len(service_df) * 35),
        template='plotly_white',
        showlegend=False
    )

    return fig


def create_fiscal_year_forecast_chart(df: pd.DataFrame, fy_label: str = "Fiscal Year") -> go.Figure:
    """
    Create a bar chart showing fiscal year forecast with actual vs projected spend.

    Actual months are shown in blue, forecasted months in orange with pattern.
    Includes prediction intervals if available.

    Args:
        df: DataFrame with columns: month_label, cost, type, prediction_low, prediction_high
        fy_label: Label for the fiscal year (e.g., "FY2025/2026")

    Returns:
        Plotly Figure object
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No forecast data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400, template='plotly_white')
        return fig

    fig = go.Figure()

    # Separate actual and forecast data
    actual_df = df[df['type'] == 'actual']
    forecast_df = df[df['type'] == 'forecast']

    # Add actual spend bars
    if not actual_df.empty:
        fig.add_trace(go.Bar(
            x=actual_df['month_label'],
            y=actual_df['cost'],
            name='Actual Spend',
            marker_color='#1f77b4',
            hovertemplate='<b>%{x}</b><br>Actual: $%{y:,.2f}<extra></extra>',
            text=[f"${v:,.0f}" for v in actual_df['cost']],
            textposition='outside'
        ))

    # Add forecast bars
    if not forecast_df.empty:
        fig.add_trace(go.Bar(
            x=forecast_df['month_label'],
            y=forecast_df['cost'],
            name='Forecast',
            marker_color='#ff7f0e',
            marker_pattern_shape='/',
            hovertemplate='<b>%{x}</b><br>Forecast: $%{y:,.2f}<extra></extra>',
            text=[f"${v:,.0f}" for v in forecast_df['cost']],
            textposition='outside'
        ))

        # Add prediction intervals if available
        if 'prediction_low' in forecast_df.columns and 'prediction_high' in forecast_df.columns:
            has_intervals = forecast_df['prediction_low'].notna().any()
            if has_intervals:
                # Add error bars for prediction intervals
                fig.add_trace(go.Scatter(
                    x=forecast_df['month_label'],
                    y=forecast_df['cost'],
                    mode='markers',
                    marker=dict(size=1, color='rgba(0,0,0,0)'),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=forecast_df['prediction_high'] - forecast_df['cost'],
                        arrayminus=forecast_df['cost'] - forecast_df['prediction_low'],
                        color='rgba(255,127,14,0.4)',
                        thickness=2,
                        width=6
                    ),
                    name='Prediction Interval',
                    showlegend=True,
                    hoverinfo='skip'
                ))

    # Calculate totals for subtitle
    total_actual = actual_df['cost'].sum() if not actual_df.empty else 0
    total_forecast = forecast_df['cost'].sum() if not forecast_df.empty else 0
    total_projected = total_actual + total_forecast

    fig.update_layout(
        title=dict(
            text=f'{fy_label} Cost Forecast (April to April)<br>'
                 f'<sub>Actual: ${total_actual:,.0f} | Forecast: ${total_forecast:,.0f} | '
                 f'Total Projected: ${total_projected:,.0f}</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Month',
        yaxis_title='Cost ($)',
        barmode='group',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='x unified'
    )

    # Add vertical line to separate actual from forecast
    if not actual_df.empty and not forecast_df.empty:
        last_actual_month = actual_df['month_label'].iloc[-1]
        all_months = df['month_label'].tolist()
        if last_actual_month in all_months:
            idx = all_months.index(last_actual_month)
            fig.add_vline(
                x=idx + 0.5,
                line_dash="dash",
                line_color="gray",
                annotation_text="Today",
                annotation_position="top"
            )

    return fig


# =====================================================================
# MULTI-ACCOUNT VISUALIZATION FUNCTIONS
# =====================================================================

def create_account_cost_bar_chart(data: Dict[str, Any], title: str = "EC2 Cost by Account") -> go.Figure:
    """
    Create horizontal bar chart showing EC2 costs by AWS account.

    Args:
        data: Multi-account cost report data
        title: Chart title

    Returns:
        Plotly Figure object
    """
    df = parse_account_costs(data)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No account cost data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400, template='plotly_white')
        return fig

    # Create color scale based on cost (darker = higher cost)
    colors = px.colors.sequential.Blues[2:]  # Skip lightest colors
    max_cost = df['total_cost'].max()
    bar_colors = [colors[min(int((c / max_cost) * (len(colors) - 1)), len(colors) - 1)]
                  for c in df['total_cost']]

    fig = go.Figure(go.Bar(
        y=df['account_name'],
        x=df['total_cost'],
        orientation='h',
        text=df.apply(lambda row: f"${row['total_cost']:,.0f} ({row['percentage']}%)", axis=1),
        textposition='outside',
        marker_color=bar_colors,
        hovertemplate='<b>%{y}</b><br>Cost: $%{x:,.2f}<br>Account ID: %{customdata}<extra></extra>',
        customdata=df['account_id']
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Cost ($)',
        yaxis=dict(autorange='reversed'),
        height=max(400, len(df) * 50),
        template='plotly_white',
        showlegend=False
    )

    return fig


def create_account_cost_pie_chart(data: Dict[str, Any], title: str = "Cost Distribution by Account") -> go.Figure:
    """
    Create donut chart showing cost distribution across accounts.

    Args:
        data: Multi-account cost report data
        title: Chart title

    Returns:
        Plotly Figure object
    """
    df = parse_account_costs(data)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No account cost data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400, template='plotly_white')
        return fig

    total_cost = df['total_cost'].sum()

    fig = go.Figure(data=[go.Pie(
        labels=df['account_name'],
        values=df['total_cost'],
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=500,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', yanchor='middle', y=0.5),
        annotations=[dict(
            text=f'Total<br>${total_cost:,.0f}',
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )]
    )

    return fig


def create_multi_account_monthly_trend(data: Dict[str, Any], title: str = "Monthly Cost Trend by Account") -> go.Figure:
    """
    Create stacked area chart showing monthly costs per account over time.

    Args:
        data: Multi-account cost report data
        title: Chart title

    Returns:
        Plotly Figure object
    """
    account_reports = data.get('account_reports', {})

    if not account_reports:
        fig = go.Figure()
        fig.add_annotation(
            text="No account data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400, template='plotly_white')
        return fig

    # Build DataFrame with all accounts' monthly costs
    all_data = []
    for account_id, account_data in account_reports.items():
        account_name = account_data.get('account_name', account_id)
        monthly_costs = account_data.get('monthly_costs', [])
        for month in monthly_costs:
            all_data.append({
                'account_name': account_name,
                'period': month.get('period', ''),
                'cost': month.get('cost', 0)
            })

    if not all_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No monthly cost data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400, template='plotly_white')
        return fig

    df = pd.DataFrame(all_data)
    df['period_label'] = df['period'].apply(lambda x: x.split(' to ')[0][:7] if ' to ' in str(x) else str(x)[:7])

    # Pivot to get accounts as columns
    pivot_df = df.pivot_table(index='period_label', columns='account_name', values='cost', aggfunc='sum').fillna(0)

    # Sort columns by total cost (highest first)
    col_totals = pivot_df.sum().sort_values(ascending=False)
    pivot_df = pivot_df[col_totals.index]

    fig = go.Figure()

    # Use a color palette
    colors = px.colors.qualitative.Set2

    for i, account in enumerate(pivot_df.columns):
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[account],
            name=account,
            mode='lines',
            stackgroup='one',
            line=dict(width=0.5),
            fillcolor=colors[i % len(colors)],
            hovertemplate=f'<b>{account}</b><br>%{{x}}<br>${{y:,.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Month',
        yaxis_title='Cost ($)',
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    return fig


def create_multi_account_kpi_cards(data: Dict[str, Any]) -> go.Figure:
    """
    Create KPI cards for multi-account overview.

    Shows 5 indicators: # Accounts, Total EC2 Cost, Total AWS Cost, EC2 %, Optimization Potential

    Args:
        data: Multi-account cost report data

    Returns:
        Plotly Figure with 5 KPI indicators
    """
    from plotly.subplots import make_subplots

    aggregated = data.get('aggregated', {})
    accounts_analyzed = data.get('accounts_analyzed', 0)

    total_ec2_cost = aggregated.get('total_ec2_cost', 0)
    total_aws_cost = aggregated.get('total_aws_cost', total_ec2_cost)
    ec2_percentage = (total_ec2_cost / total_aws_cost * 100) if total_aws_cost > 0 else 0
    optimization_potential = aggregated.get('total_optimization_potential', 0)

    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=(
            "Accounts Analyzed",
            "Total EC2 Cost (3-mo)",
            "Total AWS Cost (3-mo)",
            "EC2 % of Total",
            "Optimization Potential"
        ),
        specs=[[{"type": "indicator"}] * 5]
    )

    # Accounts Analyzed
    fig.add_trace(go.Indicator(
        mode="number",
        value=accounts_analyzed,
        number={'valueformat': "d", 'font': {'size': 48, 'color': '#1f77b4'}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=1)

    # Total EC2 Cost
    fig.add_trace(go.Indicator(
        mode="number",
        value=total_ec2_cost,
        number={'prefix': "$", 'valueformat': ",.0f", 'font': {'size': 36, 'color': '#1f77b4'}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=2)

    # Total AWS Cost
    fig.add_trace(go.Indicator(
        mode="number",
        value=total_aws_cost,
        number={'prefix': "$", 'valueformat': ",.0f", 'font': {'size': 36, 'color': '#ff7f0e'}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=3)

    # EC2 Percentage
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=ec2_percentage,
        number={'suffix': "%", 'valueformat': ".1f"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [{'range': [0, 100], 'color': "lightgray"}]
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=4)

    # Optimization Potential
    fig.add_trace(go.Indicator(
        mode="number",
        value=optimization_potential,
        number={'prefix': "$", 'valueformat': ",.0f", 'suffix': "/mo", 'font': {'size': 32, 'color': '#2ca02c'}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=5)

    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(t=60, b=20)
    )

    return fig


def create_multi_account_savings_waterfall(data: Dict[str, Any], title: str = "Optimization Potential by Account") -> go.Figure:
    """
    Create waterfall chart showing optimization potential by account.

    Args:
        data: Multi-account cost report data
        title: Chart title

    Returns:
        Plotly Figure object
    """
    account_reports = data.get('account_reports', {})

    if not account_reports:
        fig = go.Figure()
        fig.add_annotation(
            text="No account data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400, template='plotly_white')
        return fig

    # Extract savings from each account
    savings_data = []
    for account_id, account_data in account_reports.items():
        account_name = account_data.get('account_name', account_id)
        action_plan = account_data.get('action_plan', {})
        total_savings = action_plan.get('total_savings', 0)
        if total_savings > 0:
            savings_data.append({
                'account_name': account_name,
                'savings': total_savings
            })

    if not savings_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No optimization opportunities identified",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400, template='plotly_white')
        return fig

    # Sort by savings descending
    savings_data.sort(key=lambda x: x['savings'], reverse=True)

    categories = [d['account_name'] for d in savings_data] + ['Total']
    values = [d['savings'] for d in savings_data]
    total = sum(values)
    values.append(total)

    measures = ['relative'] * len(savings_data) + ['total']

    fig = go.Figure(go.Waterfall(
        name="Savings",
        orientation="v",
        measure=measures,
        x=categories,
        textposition="outside",
        text=[f"${v:,.0f}" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ca02c"}},
        totals={"marker": {"color": "#1f77b4"}}
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        height=450,
        template='plotly_white',
        yaxis_title='Monthly Savings ($)'
    )

    return fig


def create_fiscal_year_summary_kpis(forecast_data: Dict[str, Any]) -> go.Figure:
    """
    Create KPI cards for fiscal year forecast summary.

    Args:
        forecast_data: Dictionary from parse_fiscal_year_forecast()

    Returns:
        Plotly Figure with 4 KPI indicators
    """
    from plotly.subplots import make_subplots

    total_actual = forecast_data.get('total_actual', 0)
    total_forecast = forecast_data.get('total_forecast', 0)
    total_projected = forecast_data.get('total_projected', 0)
    fy_label = forecast_data.get('fiscal_year_label', 'FY')

    # Calculate progress through fiscal year
    monthly_df = forecast_data.get('monthly_df', pd.DataFrame())
    if not monthly_df.empty:
        actual_months = len(monthly_df[monthly_df['type'] == 'actual'])
        total_months = len(monthly_df)
        progress_pct = (actual_months / total_months * 100) if total_months > 0 else 0
    else:
        progress_pct = 0

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=(
            "Actual YTD",
            "Forecasted Remaining",
            f"Projected {fy_label} Total",
            "FY Progress"
        ),
        specs=[[{"type": "indicator"}, {"type": "indicator"},
                {"type": "indicator"}, {"type": "indicator"}]]
    )

    fig.add_trace(go.Indicator(
        mode="number",
        value=total_actual,
        number={'prefix': "$", 'valueformat': ",.0f"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="number",
        value=total_forecast,
        number={'prefix': "$", 'valueformat': ",.0f"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="number",
        value=total_projected,
        number={'prefix': "$", 'valueformat': ",.0f"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=3)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=progress_pct,
        number={'suffix': "%", 'valueformat': ".0f"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 100], 'color': "lightgray"}
            ]
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=4)

    fig.update_layout(
        height=250,
        showlegend=False,
        margin=dict(t=50, b=10)
    )

    return fig


# =====================================================================
# AWS CLI COMMAND GENERATION
# =====================================================================

def generate_stop_instance_command(instance_id: str, region: str = "eu-west-2") -> str:
    """Generate AWS CLI command to stop an instance."""
    return f"aws ec2 stop-instances --instance-ids {instance_id} --region {region}"


def generate_terminate_instance_command(instance_id: str, region: str = "eu-west-2") -> str:
    """Generate AWS CLI command to terminate an instance."""
    return f"aws ec2 terminate-instances --instance-ids {instance_id} --region {region}"


def generate_delete_volume_command(volume_id: str, region: str = "eu-west-2") -> str:
    """Generate AWS CLI command to delete a volume."""
    return f"aws ec2 delete-volume --volume-id {volume_id} --region {region}"


def generate_modify_volume_command(volume_id: str, new_type: str = "gp3",
                                   region: str = "eu-west-2") -> str:
    """Generate AWS CLI command to modify volume type."""
    return f"aws ec2 modify-volume --volume-id {volume_id} --volume-type {new_type} --region {region}"


def generate_console_link(instance_id: str, region: str = "eu-west-2") -> str:
    """Generate AWS Console link for an instance."""
    return f"https://{region}.console.aws.amazon.com/ec2/v2/home?region={region}#InstanceDetails:instanceId={instance_id}"


def generate_batch_stop_script(instance_ids: List[str], region: str = "eu-west-2") -> str:
    """Generate shell script to stop multiple instances."""
    commands = [
        "#!/bin/bash",
        "# Batch stop EC2 instances",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "set -e  # Exit on error",
        "",
        "echo 'Stopping instances...'",
        ""
    ]

    for instance_id in instance_ids:
        commands.append(f"aws ec2 stop-instances --instance-ids {instance_id} --region {region}")
        commands.append(f"echo 'Stopped {instance_id}'")
        commands.append("")

    commands.append("echo 'Done!'")

    return "\n".join(commands)


# =====================================================================
# FORMATTING UTILITIES
# =====================================================================

def format_currency(amount: float) -> str:
    """Format number as currency with commas."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format number as percentage."""
    return f"{value:.1f}%"


def format_large_number(value: float) -> str:
    """Format large numbers with K/M suffixes."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.2f}"


def truncate_string(s: str, max_length: int = 30) -> str:
    """Truncate string with ellipsis if too long."""
    return s if len(s) <= max_length else s[:max_length-3] + "..."


# =====================================================================
# TABLE STYLING
# =====================================================================

def style_dataframe(df: pd.DataFrame, money_columns: List[str] = None,
                   percent_columns: List[str] = None) -> pd.DataFrame:
    """Apply formatting to DataFrame for display."""
    styled_df = df.copy()

    if money_columns:
        for col in money_columns:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(format_currency)

    if percent_columns:
        for col in percent_columns:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(format_percentage)

    return styled_df
