#!/usr/bin/env python3
############################################################ IDENT(1)
#
# $Title: bpydot - Python Code Analysis and Visualization Tool $
# $Copyright: 2025 Devin Teske. All rights reserved. $
# pylint: disable=line-too-long
# $FrauBSD: bpydot/bpydot/bpydot.py 2026-01-12 10:54:21 -0800 freebsdfrau $
# pylint: enable=line-too-long
#
############################################################ LICENSE
#
# BSD 2-Clause
#
############################################################ DOCSTRING

"""
Standalone tool for analyzing Python codebases and generating call graphs, API
documentation, and architecture visualizations.

Inspired by FreeBSD bsdconfig's API module (written by Devin Teske).
"""

############################################################ IMPORTS

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from bpydot import analyze_codebase, VERSION, VERSION_VERBOSE
# pylint: enable=wrong-import-position

############################################################ FUNCTIONS

def main():
    """Main entry point for bpydot CLI."""
    parser = argparse.ArgumentParser(
        prog='bpydot',
        description='Python code analysis and visualization tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze current directory, high detail, output to stdout
  bpydot.py

  # Analyze specific project
  bpydot.py /path/to/project

  # Generate dot file with medium detail
  bpydot.py -l medium -o api.dot /path/to/project

  # Story-driven view (low detail, key functions only)
  bpydot.py -l low -o overview.dot

  # Generate SVG directly (requires graphviz)
  bpydot.py -o api.dot && dot -Tsvg api.dot -o api.svg

Use Cases:
  - Pre-flight PR review (identify vestigial code, duplicates)
  - Architectural exploration (understand large codebases)
  - API documentation (advertise available functions)
  - Technical debt analysis (unused globals, orphan functions)
""",
    )

    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Path to Python project root (default: current directory)',
    )

    parser.add_argument(
        '-l', '--level',
        choices=['low', 'medium', 'high'],
        default='high',
        help='Detail level: low (story-driven, key functions),'
			+ ' medium (public APIs + phases),'
			+ ' high (all functions + globals + signatures, default)',
    )

    parser.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Write output to file (default: stdout)',
    )

    parser.add_argument(
        '-m', '--module',
        metavar='MODULE',
        help='Filter to specific module (e.g., "analysis.jobs", "io")',
    )

    parser.add_argument(
        '-v', '--version',
        action='store_true',
        help='Show version information and exit',
    )

    parser.add_argument(
        '-V', '--version-verbose',
        action='store_true',
        help='Show verbose version with build info and exit',
    )

    args = parser.parse_args()

    # Handle version flags
    if args.version_verbose:
        print(VERSION_VERBOSE)
        return 0

    if args.version:
        print(VERSION)
        return 0

    # Validate path
    root_path = Path(args.path).resolve()
    if not root_path.exists():
        print(f"Error: Path does not exist: {root_path}", file=sys.stderr)
        return 1

    if not root_path.is_dir():
        print(f"Error: Path is not a directory: {root_path}", file=sys.stderr)
        return 1

    # Check for Python files
    py_files = list(root_path.rglob('*.py'))
    if not py_files:
        print(f"Error: No Python files found in: {root_path}", file=sys.stderr)
        return 1

    # Run analysis
    try:
        analyze_codebase(
            root_path=str(root_path),
            level=args.level,
            module_filter=args.module,
            output_file=args.output,
        )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit()
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error: {e}", file=sys.stderr)
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()
        sys.exit()

############################################################ MAIN

if __name__ == '__main__':
    sys.exit(main())

################################################################################
# END
################################################################################
