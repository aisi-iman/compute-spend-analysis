#!/usr/bin/env python3
"""
AWS Multi-Account Setup Script

Automatically discovers AWS accounts via SSO and generates configuration files
for multi-account cost analysis.

Usage:
    uv run python setup_multi_account.py
"""

import boto3
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_sso_token_from_cache() -> Optional[str]:
    """
    Read SSO access token from AWS CLI cache.

    Returns:
        Access token string if found and valid, None otherwise.
    """
    cache_dir = Path.home() / ".aws" / "sso" / "cache"

    if not cache_dir.exists():
        return None

    # Find the most recent cache file
    cache_files = list(cache_dir.glob("*.json"))
    if not cache_files:
        return None

    # Sort by modification time, newest first
    cache_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    for cache_file in cache_files:
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Check if this is an SSO token (not a client registration)
            if 'accessToken' in data:
                # Check expiration
                expires_at = data.get('expiresAt', '')
                if expires_at:
                    from datetime import datetime
                    # Parse ISO format: 2025-12-23T12:00:00Z
                    try:
                        exp_time = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                        if exp_time > datetime.now(exp_time.tzinfo):
                            return data['accessToken']
                    except:
                        # If we can't parse, try using the token anyway
                        return data['accessToken']
        except (json.JSONDecodeError, KeyError):
            continue

    return None


def run_sso_login(sso_start_url: str, sso_region: str) -> bool:
    """
    Run AWS SSO login via CLI.

    Args:
        sso_start_url: SSO portal URL
        sso_region: AWS region for SSO

    Returns:
        True if login succeeded
    """
    print("\nInitiating SSO login...")
    print("A browser window will open for authentication.\n")

    # Create a temporary SSO session config
    session_name = "cost-analyzer-setup"

    # First, configure the SSO session
    config_path = Path.home() / ".aws" / "config"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing config
    existing_config = ""
    if config_path.exists():
        existing_config = config_path.read_text()

    # Add SSO session if not present
    sso_session_block = f"""
[sso-session {session_name}]
sso_start_url = {sso_start_url}
sso_region = {sso_region}
sso_registration_scopes = sso:account:access
"""

    if f"[sso-session {session_name}]" not in existing_config:
        with open(config_path, 'a') as f:
            f.write(sso_session_block)

    # Run SSO login
    try:
        result = subprocess.run(
            ["aws", "sso", "login", "--sso-session", session_name],
            check=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"SSO login failed: {e}")
        return False
    except FileNotFoundError:
        print("Error: AWS CLI not found. Please install AWS CLI v2.")
        return False


def list_sso_accounts(access_token: str) -> List[Dict]:
    """
    List all AWS accounts accessible via SSO.

    Args:
        access_token: SSO access token

    Returns:
        List of account dictionaries with accountId, accountName, emailAddress
    """
    sso_client = boto3.client('sso', region_name='eu-west-2')

    accounts = []
    paginator = sso_client.get_paginator('list_accounts')

    for page in paginator.paginate(accessToken=access_token):
        accounts.extend(page.get('accountList', []))

    return accounts


def list_account_roles(access_token: str, account_id: str) -> List[Dict]:
    """
    List available roles for an account.

    Args:
        access_token: SSO access token
        account_id: AWS account ID

    Returns:
        List of role dictionaries with roleName, accountId
    """
    sso_client = boto3.client('sso', region_name='eu-west-2')

    roles = []
    paginator = sso_client.get_paginator('list_account_roles')

    try:
        for page in paginator.paginate(accessToken=access_token, accountId=account_id):
            roles.extend(page.get('roleList', []))
    except Exception as e:
        print(f"  Warning: Could not list roles for account {account_id}: {e}")

    return roles


def generate_aws_config(
    accounts: List[Tuple[str, str, str, str]],  # (account_id, account_name, role_name, profile_name)
    sso_start_url: str,
    sso_region: str,
    default_region: str = "eu-west-2"
) -> str:
    """
    Generate AWS CLI config file content.

    Returns:
        Config file content as string
    """
    session_name = "cost-analyzer-setup"

    config_lines = [
        "# AWS CLI Configuration",
        "# Auto-generated by setup_multi_account.py",
        "",
        f"[sso-session {session_name}]",
        f"sso_start_url = {sso_start_url}",
        f"sso_region = {sso_region}",
        "sso_registration_scopes = sso:account:access",
        ""
    ]

    for account_id, account_name, role_name, profile_name in accounts:
        config_lines.extend([
            f"[profile {profile_name}]",
            f"sso_session = {session_name}",
            f"sso_account_id = {account_id}",
            f"sso_role_name = {role_name}",
            f"region = {default_region}",
            f"# Account: {account_name}",
            ""
        ])

    return "\n".join(config_lines)


def generate_accounts_config(
    accounts: List[Tuple[str, str, str, str]]  # (account_id, account_name, role_name, profile_name)
) -> str:
    """
    Generate accounts_config.yaml content.

    Returns:
        YAML config content as string
    """
    lines = [
        "# Multi-Account Cost Analyzer Configuration",
        "# Auto-generated by setup_multi_account.py",
        "",
        "accounts:"
    ]

    for account_id, account_name, role_name, profile_name in accounts:
        # Sanitize account name for YAML
        safe_name = account_name.replace('"', '\\"')
        lines.extend([
            f'  - profile: "{profile_name}"',
            f'    name: "{safe_name}"',
            f'    account_id: "{account_id}"',
            f'    enabled: true',
            ""
        ])

    lines.extend([
        "settings:",
        "  # Continue analyzing other accounts if one fails",
        "  continue_on_error: true",
        "  # Include accounts with zero EC2 spend in report",
        "  include_zero_spend_accounts: false",
        ""
    ])

    return "\n".join(lines)


def sanitize_profile_name(account_name: str, account_id: str) -> str:
    """Create a valid AWS profile name from account name."""
    # Remove special characters, convert spaces to hyphens
    safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in account_name.lower())
    # Remove consecutive hyphens
    while "--" in safe_name:
        safe_name = safe_name.replace("--", "-")
    # Trim and ensure not empty
    safe_name = safe_name.strip("-")
    if not safe_name:
        safe_name = f"account-{account_id[-4:]}"
    return f"aisi-{safe_name}"


def main():
    print("=" * 60)
    print("AWS Multi-Account Setup for Cost Analyzer")
    print("=" * 60)
    print()
    print("This script will:")
    print("  1. Authenticate via AWS SSO")
    print("  2. Discover all accounts you have access to")
    print("  3. Generate AWS CLI profiles for each account")
    print("  4. Generate accounts_config.yaml for the cost analyzer")
    print()

    # Get SSO URL
    sso_start_url = input("Enter your AWS SSO start URL\n(e.g., https://myorg.awsapps.com/start): ").strip()

    if not sso_start_url:
        print("Error: SSO URL is required")
        sys.exit(1)

    # Normalize URL
    if not sso_start_url.startswith("https://"):
        sso_start_url = f"https://{sso_start_url}"

    sso_region = input("\nEnter your SSO region [eu-west-2]: ").strip() or "eu-west-2"
    default_region = input("Enter default AWS region for analysis [eu-west-2]: ").strip() or "eu-west-2"

    # Try to get existing token or login
    print("\nChecking for existing SSO session...")
    access_token = get_sso_token_from_cache()

    if not access_token:
        print("No valid SSO session found. Starting login...")
        if not run_sso_login(sso_start_url, sso_region):
            print("Error: SSO login failed")
            sys.exit(1)

        # Wait a moment for cache to be written
        time.sleep(2)
        access_token = get_sso_token_from_cache()

        if not access_token:
            print("Error: Could not obtain SSO access token after login")
            sys.exit(1)

    print("SSO session active!")

    # Discover accounts
    print("\nDiscovering AWS accounts...")
    try:
        accounts = list_sso_accounts(access_token)
    except Exception as e:
        print(f"Error listing accounts: {e}")
        sys.exit(1)

    if not accounts:
        print("No accounts found. Check your SSO permissions.")
        sys.exit(1)

    print(f"\nFound {len(accounts)} accounts:")
    print("-" * 60)

    # Discover roles for each account
    account_configs = []

    for account in accounts:
        account_id = account['accountId']
        account_name = account.get('accountName', account_id)
        email = account.get('emailAddress', 'N/A')

        print(f"\n  Account: {account_name}")
        print(f"    ID: {account_id}")
        print(f"    Email: {email}")

        # Get available roles
        roles = list_account_roles(access_token, account_id)

        if not roles:
            print("    Roles: None found (skipping)")
            continue

        print(f"    Roles: {', '.join(r['roleName'] for r in roles)}")

        # Only use specific roles that have the required permissions
        allowed_roles = ['AISI-Research-Platform-Extended', 'AWSAdministratorAccess']
        available_allowed = [r['roleName'] for r in roles if r['roleName'] in allowed_roles]

        if not available_allowed:
            print(f"    No suitable role found (need one of: {', '.join(allowed_roles)}) - skipping")
            continue

        # Prefer AISI-Research-Platform-Extended over AWSAdministratorAccess
        if 'AISI-Research-Platform-Extended' in available_allowed:
            selected_role = 'AISI-Research-Platform-Extended'
        else:
            selected_role = available_allowed[0]

        print(f"    Selected: {selected_role}")

        profile_name = sanitize_profile_name(account_name, account_id)
        account_configs.append((account_id, account_name, selected_role, profile_name))

    if not account_configs:
        print("\nNo accounts with accessible roles found.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Configuration Summary: {len(account_configs)} accounts")
    print("=" * 60)

    # Generate AWS config
    aws_config_content = generate_aws_config(
        account_configs, sso_start_url, sso_region, default_region
    )

    # Generate accounts_config.yaml
    accounts_yaml_content = generate_accounts_config(account_configs)

    # Show preview
    print("\n--- Preview: ~/.aws/config ---")
    print(aws_config_content[:1000])
    if len(aws_config_content) > 1000:
        print("... (truncated)")

    print("\n--- Preview: accounts_config.yaml ---")
    print(accounts_yaml_content)

    # Confirm before writing
    print()
    confirm = input("Write these configuration files? [Y/n]: ").strip().lower()

    if confirm in ('', 'y', 'yes'):
        # Write AWS config
        aws_config_path = Path.home() / ".aws" / "config"
        aws_config_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing config
        if aws_config_path.exists():
            backup_path = aws_config_path.with_suffix('.config.backup')
            print(f"\nBacking up existing config to: {backup_path}")
            aws_config_path.rename(backup_path)

        aws_config_path.write_text(aws_config_content)
        print(f"Wrote: {aws_config_path}")

        # Write accounts_config.yaml
        accounts_yaml_path = Path("accounts_config.yaml")
        accounts_yaml_path.write_text(accounts_yaml_content)
        print(f"Wrote: {accounts_yaml_path.absolute()}")

        print("\n" + "=" * 60)
        print("Setup complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Login to SSO: aws sso login --sso-session cost-analyzer-setup")
        print("  2. Run analysis: uv run python multi_account_analyzer.py")
        print()
    else:
        print("\nCancelled. No files written.")


if __name__ == "__main__":
    main()
