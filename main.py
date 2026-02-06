#!/usr/bin/env python3
"""
KSB Coursework Marker - Main Entry Point

Usage:
    # Run the Streamlit UI
    python main.py ui
    
    # Or directly
    streamlit run ui/ksb_app.py
"""
import argparse
import subprocess
from pathlib import Path


def run_ui():
    """Launch the Streamlit UI."""
    ui_path = Path(__file__).parent / "ui" / "ksb_app.py"
    subprocess.run(["streamlit", "run", str(ui_path)])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="KSB Coursework Marker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run the web UI
    python main.py ui
    
    # Or directly
    streamlit run ui/ksb_app.py
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.add_parser('ui', help='Launch Streamlit UI')
    
    args = parser.parse_args()
    
    if args.command == 'ui':
        run_ui()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
