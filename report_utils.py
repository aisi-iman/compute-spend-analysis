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
