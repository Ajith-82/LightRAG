#!/usr/bin/env python3
"""
xAI Configuration Validation and Optimization Tool.

This script helps validate and optimize xAI configuration for LightRAG.
It can be run standalone to check configuration and apply optimizations.
"""

import argparse
import os
import sys
from pathlib import Path

# Add LightRAG to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.xai_optimizer import (
    XAIConfigOptimizer,
    optimize_xai_config,
    validate_xai_config,
    generate_xai_env_config,
    print_xai_optimization_report
)


def load_env_file(env_path: str = ".env"):
    """Load environment variables from .env file."""
    env_file = Path(env_path)
    if not env_file.exists():
        print(f"❌ Environment file not found: {env_path}")
        return False
    
    # Simple .env parser
    with open(env_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    os.environ[key] = value
                except ValueError:
                    print(f"⚠️  Warning: Invalid line {line_num} in {env_path}: {line}")
    
    print(f"✅ Loaded environment from {env_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate and optimize xAI configuration for LightRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --validate                 # Validate current configuration
  %(prog)s --optimize                 # Show optimization recommendations
  %(prog)s --optimize --apply         # Apply optimizations to environment
  %(prog)s --generate grok-3-mini     # Generate optimal .env config
  %(prog)s --report                   # Show detailed report
  %(prog)s --env custom.env --report  # Use custom .env file
        """
    )
    
    parser.add_argument(
        "--env", 
        default=".env",
        help="Path to .env file (default: .env)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate current xAI configuration"
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true", 
        help="Show optimization recommendations"
    )
    
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply optimizations (use with --optimize)"
    )
    
    parser.add_argument(
        "--generate",
        metavar="MODEL",
        help="Generate optimal .env config for specified model"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show detailed optimization report"
    )
    
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Output file for generated config (use with --generate)"
    )
    
    args = parser.parse_args()
    
    # Load environment file if exists
    if os.path.exists(args.env):
        if not load_env_file(args.env):
            sys.exit(1)
    else:
        print(f"⚠️  Environment file not found: {args.env}")
        if not any([args.generate, args.help]):
            print("   Continuing with system environment variables only...")
    
    # Handle different operations
    if args.generate:
        print(f"🔧 Generating optimal configuration for {args.generate}")
        config = generate_xai_env_config(args.generate)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(config)
            print(f"✅ Configuration written to {args.output}")
        else:
            print("\n" + "="*60)
            print(config)
            print("="*60)
        
        return
    
    if args.report:
        print_xai_optimization_report()
        return
    
    if args.validate:
        print("🔍 Validating xAI configuration...")
        is_valid, issues = validate_xai_config()
        
        if is_valid:
            print("✅ Configuration is valid and optimal")
        else:
            print(f"❌ Found {len(issues)} configuration issues:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        
        return
    
    if args.optimize:
        print("🔧 Analyzing configuration for optimizations...")
        result = optimize_xai_config(apply_changes=args.apply)
        
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        
        if 'issues' in result and result['issues']:
            print(f"\nIssues found:")
            for issue in result['issues']:
                print(f"  - {issue}")
        
        if 'changes' in result and result['changes']:
            action = "Applied" if args.apply else "Recommended"
            print(f"\n{action} changes:")
            for key, value in result['changes'].items():
                print(f"  {key}={value}")
        
        if not args.apply and 'changes' in result and result['changes']:
            print(f"\n💡 To apply these changes, run:")
            print(f"   {sys.argv[0]} --optimize --apply")
        
        return
    
    # Default: show help if no specific action
    parser.print_help()


if __name__ == "__main__":
    main()