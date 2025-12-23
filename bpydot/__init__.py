############################################################ IDENT(1)
#
# $Title: Python init for bpydot package $
# $Copyright: 2025 Devin Teske. All rights reserved. $
# pylint: disable=line-too-long
# $FrauBSD$
# pylint: enable=line-too-long
#
############################################################ LICENSE
#
# BSD 2-Clause
#
############################################################ DOCSTRING

"""bpydot: Python code analysis and visualization tool.

Inspired by FreeBSD bsdconfig's API module, bpydot provides:
- AST-based code introspection
- Call graph generation (Graphviz dot format)
- Global variable tracking
- Multiple detail levels (low/medium/high)

Example:
    from bpydot import analyze_codebase

    analyze_codebase(
        root_path='/path/to/project',
        level='high',
        output_file='api.dot',
    )
"""

############################################################ IMPORTS

from .analyzer import run_api_introspection as analyze_codebase
from .version import VERSION, VERSION_VERBOSE

############################################################ SETUP

__version__ = VERSION
__all__ = ['analyze_codebase', 'VERSION', 'VERSION_VERBOSE']

################################################################################
# END
################################################################################
