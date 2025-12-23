#!/usr/bin/env python3
############################################################ IDENT(1)
#
# $Title: bpydot - Python Code Analysis and Visualization $
# $Copyright: 2025 Devin Teske. All rights reserved. $
# pylint: disable=line-too-long
# $FrauBSD: bpydot/bpydot/analyzer.py 2025-12-22 20:51:35 -0800 freebsdfrau $
# pylint: enable=line-too-long
# pylint: disable=too-many-lines,too-many-instance-attributes,too-few-public-methods
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-nested-blocks,too-many-return-statements
# pylint: disable=broad-exception-caught
#
############################################################ LICENSE
#
# BSD 2-Clause
#
############################################################ DOCSTRING

"""This module provides real-time code analysis and visualization capabilities.
Inspired by FreeBSD bsdconfig's API module (written by Devin Teske), this
analyzes Python codebases to:

1. Advertise available API functions (avoid reinventing the wheel)
2. Visualize program architecture (understand the structure)
3. Generate call graphs (trace execution flow)
4. Explore as you build (real-time, no manual updates)

Leverages Python's AST for deep static analysis.
"""

############################################################ IMPORTS

import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from .version import VERSION, VERSION_VERBOSE
from .utils import plur

############################################################ GLOBALS

# Module color scheme (visually appealing palette)
MODULE_COLORS = {
    'analysis': '#E3F2FD',      # Light blue - data analysis
    'io': '#FFF3E0',            # Light orange - I/O operations
    'utils': '#F3E5F5',         # Light purple - utilities
    'config': '#E8F5E9',        # Light green - configuration
    'cli': '#FCE4EC',           # Light pink - user interface
    'introspection': '#FFF9C4', # Light yellow - meta (this module!)
}

# Darker versions for edges (visible on white background)
EDGE_COLORS = {
    'analysis': '#1976D2',      # Blue
    'io': '#F57C00',            # Orange
    'utils': '#7B1FA2',         # Purple
    'config': '#388E3C',        # Green
    'cli': '#C2185B',           # Pink
    'introspection': '#F9A825', # Yellow
    'root': '#616161',          # Gray
}

# Node shapes by function type
NODE_SHAPES = {
    'phase': 'component',       # Phase orchestrators (box with 3D effect)
    'public_api': 'box',        # Public API functions
    'internal': 'ellipse',      # Internal helpers
    'data_structure': 'folder', # Data manipulation
    'io_operation': 'cylinder', # I/O operations
    'decision': 'diamond',      # Routing/decision functions
}

# Edge styles by call type
EDGE_STYLES = {
    'direct': {'style': 'solid', 'penwidth': '2.0', 'color': '#424242'},
    'conditional': {'style': 'dashed', 'penwidth': '1.5', 'color': '#757575'},
    'error_path': {'style': 'dotted', 'penwidth': '1.0', 'color': '#D32F2F'},
    'loop': {'style': 'bold', 'penwidth': '2.5', 'color': '#1976D2'},
}

############################################################ CLASSES

class FunctionInfo:
    """Metadata about a function."""

    def __init__(self, name: str, module: str, lineno: int,
                 end_lineno: int = None):
        self.name = name
        self.module = module
        self.lineno = lineno
        self.end_lineno = end_lineno  # for accurate function source extraction
        self.docstring = None
        self.calls = []     # Functions this calls
        self.called_by = [] # Functions that call this
        self.is_public = not name.startswith('_')
        self.is_phase = 'phase' in name.lower() or 'execute' in name.lower()
        self.is_io = any(x in name.lower() for x in ['read', 'write', 'load',
                                                     'save', 'query'])
        self.is_decision = any(x in name.lower() for x in ['check',
                                                           'determine',
                                                           'validate',
                                                           'filter'])
        self.parameters = []
        self.return_type = None

    def get_node_shape(self) -> str:
        """Determine node shape based on function characteristics."""
        if self.is_phase:
            return NODE_SHAPES['phase']
        if self.is_io:
            return NODE_SHAPES['io_operation']
        if self.is_decision:
            return NODE_SHAPES['decision']
        if self.is_public:
            return NODE_SHAPES['public_api']
        return NODE_SHAPES['internal']

    def get_label(self, include_signature: bool = False,
                  include_return: bool = False) -> str:
        """Generate node label with optional signature and return type."""
        if not include_signature:
            return self.name

        if self.parameters:
            # Limit to 3 for readability
            params = ", ".join(self.parameters[:3])
            if len(self.parameters) > 3:
                params += ", ..."
            label = f"{self.name}({params})"
        else:
            # Always show () for functions, even with no args
            label = f"{self.name}()"

        # Add return type if requested and available
        if include_return and self.return_type:
            label += f" → {self.return_type}"

        return label


class ModuleInfo:
    """Metadata about a Python module."""

    def __init__(self, path: str, package: str):
        self.path = path
        self.package = package           # e.g., 'foo.bar.baz'
        self.functions = {}              # name -> FunctionInfo
        self.imports = []                # External imports
        self.globals = []                # Global variables
        self.classes = []                # Class definitions
        self.source_code = None          # Source code of the module
        self.imported_vars = set()  # Variables imported from other modules

    def get_subsystem(self) -> str:
        """Get top-level subsystem name (analysis, io, utils, etc.)."""
        parts = self.package.split('.')
        return parts[0] if len(parts) > 1 else '.'


class CallGraphAnalyzer:
    """Analyzes Python code to build call graph."""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.modules = {}           # package_name -> ModuleInfo
        self.functions = {}         # full_name -> FunctionInfo
        self.call_edges = []        # List of (caller, callee) tuples
        self.variable_mappings = {} # variable_name -> function_full_name
                                    # NB: for indirection tracking
        self.graphable_globals = set()  # Globals that should be graphed

    def analyze(self, exclude_patterns: Optional[List[str]] = None,
                skip_init_files: bool = True):
        """Walk codebase and analyze all Python files."""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'test', '.pyc']

        # Find all Python files
        python_files = []
        for path in self.root_path.rglob('*.py'):
            # Skip excluded patterns
            if any(pat in str(path) for pat in exclude_patterns):
                continue
            # Skip __init__.py files (they're usually just exports, not logic)
            if skip_init_files and path.name == '__init__.py':
                continue
            python_files.append(path)

        # Parse each file
        for py_file in python_files:
            self._analyze_file(py_file)

        # Build call graph edges
        self._build_call_graph()

        # Identify which globals are "interesting" (meet criteria 1+2)
        self._identify_graphable_globals()

    def _identify_graphable_globals(self):
        """Identify globals that should be graphed.

        Criteria:
        1. Defined at module level (outside functions/classes)
        2. Imported by other modules (via 'from X import Y' or 'import X')

        Only globals meeting BOTH criteria are marked as graphable.
        """
        self.graphable_globals = set()

        # For each module, check if its globals are imported elsewhere
        for defining_package, defining_module in self.modules.items():
            for global_name in defining_module.globals:
                # Check if ANY other module imports this global
                for importing_package, importing_module in self.modules.items():
                    if importing_package == defining_package:
                        continue  # Skip same module

                    # Check if this module imports the global
                    if hasattr(importing_module, 'imported_vars') and \
                        global_name in importing_module.imported_vars:
                        # This global is imported! Mark as graphable
                        # Store as "package.global_name" for uniqueness
                        graphable_id = f"{defining_package}.{global_name}"
                        self.graphable_globals.add(graphable_id)

    def _analyze_file(self, file_path: Path):
        """Parse a single Python file using AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))

            # Build package name from path
            rel_path = file_path.relative_to(self.root_path)
            package = str(rel_path.with_suffix('')).replace(os.sep, '.')

            module_info = ModuleInfo(str(file_path), package)
            module_info.source_code = source  # Store for variable tracking

            #
            # Extract functions, classes, imports
            # NB: Use tree.body (top-level only), NOT ast.walk()
            # NB: ast.walk() traverses info functions and catches function-
            #     local variables.
            #
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    self._extract_function(node, module_info)
                elif isinstance(node, ast.ClassDef):
                    module_info.classes.append(node.name)
                    # ALSO extract methods from classes (for global tracking)
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            self._extract_function(item, module_info)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._extract_import(node, module_info)
                elif isinstance(node, ast.Assign):
                    # ONLY module-level assignments (tree.body = top-level only)
                    # This correctly excludes function-local assignments
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id

                        # FILTER: Skip type aliases (e.g., Foo = str)
                        # These are not real runtime globals, just type hints
                        if isinstance(node.value, (ast.Name, ast.Subscript)):
                            # Check if RHS looks like a type annotation
                            # ast.Name: Foo = str (type alias)
                            # ast.Subscript: Foo = Dict[str, Any] (type alias)
                            if isinstance(node.value, ast.Name):
                                # Simple names that are common types
                                type_names = {'str', 'int', 'float', 'bool',
                                              'bytes', 'list', 'dict', 'set',
                                              'tuple', 'Any', 'Optional',
                                              'Union', 'List', 'Dict', 'Set',
                                              'Tuple'}
                                if node.value.id in type_names:
                                    continue  # Skip type alias
                            elif isinstance(node.value, ast.Subscript):
                                # Generic types like Dict[...], List[...], etc.
                                if isinstance(node.value.value, ast.Name):
                                    generic_types = {'Dict', 'List', 'Set',
                                                     'Tuple', 'Optional',
                                                     'Union', 'TypeVar'}
                                    if node.value.value.id in generic_types:
                                        continue  # Skip type alias

                        # Track ALL other top-level module variables
                        module_info.globals.append(var_name)

                        # Track if variable is assigned from a function call
                        # e.g., VERSION = get_version()
                        if isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Name):
                                # Simple function call: func()
                                func_name = node.value.func.id
                                full_name = f"{module_info.package}.{func_name}"
                                self.variable_mappings[var_name] = full_name
                elif isinstance(node, ast.AnnAssign):
                    # Handle annotated assignments: var: Type = value
                    # Common in modern Python (e.g., foo: Dict[str, int] = {})
                    if isinstance(node.target, ast.Name):
                        var_name = node.target.id
                        module_info.globals.append(var_name)

                        # Check if assigned from function call
                        if node.value and isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Name):
                                func_name = node.value.func.id
                                full_name = f"{module_info.package}.{func_name}"
                                self.variable_mappings[var_name] = full_name

            self.modules[package] = module_info

        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}", file=sys.stderr)

    def _extract_function(self, node: ast.FunctionDef, module_info: ModuleInfo):
        """Extract function metadata from AST node."""
        # Skip __init__ and other dunder methods (class internals, not API)
        if node.name.startswith('__') and node.name.endswith('__'):
            return

        end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else None
        func = FunctionInfo(node.name, module_info.package, node.lineno, end_lineno)

        # Extract docstring
        func.docstring = ast.get_docstring(node)

        # Extract parameters
        for arg in node.args.args:
            func.parameters.append(arg.arg)

        # Extract return type annotation
        if node.returns:
            # Try ast.unparse (Python 3.9+) or use simple name extraction
            if hasattr(ast, 'unparse'):
                func.return_type = ast.unparse(node.returns)
            elif isinstance(node.returns, ast.Name):
                # Simple return type: -> str
                func.return_type = node.returns.id
            elif isinstance(node.returns, ast.Constant):
                # Constant type: -> None
                func.return_type = str(node.returns.value)
            elif hasattr(node.returns, 'id'):
                # Generic Name node
                func.return_type = node.returns.id

        # Extract function calls within this function
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    # Simple function call: foo()
                    func.calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # Method call: obj.foo()
                    func.calls.append(child.func.attr)

            # NB: Track variable assignments from function calls.
            # NB: This catches foo = bar()
            elif isinstance(child, ast.Assign):
                if isinstance(child.value, ast.Call):
                    if isinstance(child.value.func, ast.Name):
                        # This is an assignment from a function call
                        called_func = child.value.func.id
                        # Add to calls list (creates edge to called function)
                        func.calls.append(called_func)

        # Store in module
        module_info.functions[node.name] = func

        # Store in global function registry
        full_name = f"{module_info.package}.{node.name}"
        self.functions[full_name] = func

    def _extract_import(self, node: ast.AST, module_info: ModuleInfo):
        """Extract import statements."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_info.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_info.imports.append(node.module)
                # Track ALL imported names (we'll filter later to find which are globals)
                # This is generic - works for any project
                for name in node.names:
                    if hasattr(name, 'name'):
                        imported_name = name.name
                        # Track ALL imports - we'll determine if they're globals later
                        if not hasattr(module_info, 'imported_vars'):
                            module_info.imported_vars = set()
                        module_info.imported_vars.add(imported_name)

    def _build_call_graph(self):
        """Build edges between functions based on calls."""
        # Use a set to deduplicate edges (multiple calls to same function = one edge)
        edge_set = set()

        for full_name, func in self.functions.items():
            # Deduplicate calls within this function
            unique_calls = set(func.calls)

            for called_func_name in unique_calls:
                # Try to resolve the called function
                # This is simplified - full resolution would require import tracking
                for other_full_name, other_func in self.functions.items():
                    if other_func.name == called_func_name:
                        # Skip self-edges (recursion or super() calls)
                        if full_name == other_full_name:
                            continue
                        edge_set.add((full_name, other_full_name))
                        other_func.called_by.append(full_name)

        # Convert set back to list (now deduplicated)
        self.call_edges = list(edge_set)

    def get_entry_points(self) -> List[str]:
        """Find entry point functions (called by no one internally)."""
        called_funcs = set(edge[1] for edge in self.call_edges)
        all_funcs = set(self.functions.keys())
        return sorted(all_funcs - called_funcs)


############################################################ FUNCTIONS

def should_show_node(func, full_name: str, level: str, subsystem: str,
                     nodes_in_edges: set) -> bool:
    """Determine if a node should be shown based on level and filters.

    Args:
        func: FunctionInfo object
        full_name: Full qualified function name
        level: Detail level (low/medium/high)
        subsystem: Subsystem name
        nodes_in_edges: Set of function names that appear in edges

    Returns:
        True if node should be shown, False otherwise
    """
    # Skip introspection module at low/medium levels (it's meta/self-referential)
    if level in ['low', 'medium'] and subsystem == 'introspection':
        return False

    # Must show if referenced in edges (BUT respect private filter at low)
    if full_name in nodes_in_edges:
        if level == 'low' and func.name.startswith('_'):
            return False
        return True

    # High level: show everything
    if level == 'high':
        return True

    # Medium level: show public, phase, io, or decision functions
    if level == 'medium' and (func.is_public or func.is_phase or
                              func.is_io or func.is_decision):
        return True

    # Low level: show public or phase functions (not private)
    if level == 'low' and (func.is_public or func.is_phase) and not \
        func.name.startswith('_'):
        return True

    return False


def get_node_color_and_style(func) -> tuple:
    """Get node color, penwidth, and style based on function type.

    Args:
        func: FunctionInfo object

    Returns:
        Tuple of (fillcolor, penwidth, style, color_suffix, node_category)
        where node_category is 'phase', 'public', or 'internal'
    """
    # Determine if private function (starts with _ but not __)
    is_private = func.name.startswith('_') and not (func.name.startswith('__') and
                                                    func.name.endswith('__'))

    # Color by function type
    if func.is_phase:
        fillcolor = '#FFE082'  # Gold for phases
        penwidth = '2.5'
        style = 'filled'
        node_category = 'phase'
    elif func.is_io:
        fillcolor = '#FFCCBC'  # Orange for I/O
        penwidth = '1.5'
        style = 'filled'
        node_category = 'internal'
    elif func.is_decision:
        fillcolor = '#B2DFDB'  # Teal for decisions
        penwidth = '1.5'
        style = 'filled'
        node_category = 'internal'
    elif func.is_public:
        fillcolor = '#C5E1A5'  # Light green for public API
        penwidth = '1.5'
        style = 'filled'
        node_category = 'public'
    else:
        fillcolor = '#F5F5F5'  # Light gray for internal
        penwidth = '1.0'
        style = 'filled'
        node_category = 'internal'

    # Private functions get dashed border
    color_suffix = ''
    if is_private:
        style = 'filled,dashed'
        color_suffix = ', color="#999999"'

    return fillcolor, penwidth, style, color_suffix, node_category


def generate_function_node_lines(func, module, level: str, indent: str = '        ') -> List[str]:
    """Generate dot language lines for a function node.

    Args:
        func: FunctionInfo object
        module: ModuleInfo object
        level: Detail level
        indent: Indentation string

    Returns:
        List of dot language strings
    """
    lines = []
    node_id = f"{module.package}_{func.name}".replace('.', '_')
    shape = func.get_node_shape()
    label = func.get_label(include_signature=level == 'high',
                           include_return=level == 'high')

    fillcolor, penwidth, style, color_suffix, _ = get_node_color_and_style(func)

    # Add docstring as tooltip
    tooltip = ''
    if func.docstring:
        first_line = func.docstring.split('\n')[0][:80]
        first_line = first_line.replace('"', '\\"')
        tooltip = f', tooltip="{first_line}"'

    lines.append(f"{indent}{node_id} [")
    lines.append(f'{indent}    label="{label}",')
    lines.append(f'{indent}    shape="{shape}",')
    lines.append(f'{indent}    style="{style}",')
    lines.append(f'{indent}    fillcolor="{fillcolor}",')
    lines.append(f'{indent}    penwidth="{penwidth}"{color_suffix}{tooltip}')
    lines.append(f'{indent}];')

    return lines


def generate_global_node_lines(global_name: str, module, analyzer: CallGraphAnalyzer,
                               indent: str = '        ') -> List[str]:
    """Generate dot language lines for a global variable node.

    Args:
        global_name: Name of the global variable
        module: ModuleInfo object
        analyzer: CallGraphAnalyzer instance
        indent: Indentation string

    Returns:
        List of dot language strings
    """
    lines = []
    global_id = f"{module.package}_{global_name}".replace('.', '_')

    if global_name in analyzer.variable_mappings:
        # Dark purple for function-assigned globals
        lines.append(f"{indent}{global_id} [")
        lines.append(f'{indent}    label="{global_name}",')
        lines.append(f'{indent}    shape="hexagon",')
        lines.append(f'{indent}    style="filled",')
        lines.append(f'{indent}    fillcolor="#E1BEE7",')
        lines.append(f'{indent}    penwidth=2.0,')
        lines.append(f'{indent}    tooltip="Global variable"')
        lines.append(f'{indent}];')
    else:
        # Light purple for literal-assigned globals
        lines.append(f"{indent}{global_id} [")
        lines.append(f'{indent}    label="{global_name}",')
        lines.append(f'{indent}    shape="hexagon",')
        lines.append(f'{indent}    style="filled",')
        lines.append(f'{indent}    fillcolor="#F3E5E5",')
        lines.append(f'{indent}    penwidth=1.5,')
        lines.append(f'{indent}    tooltip="Global variable (literal)"')
        lines.append(f'{indent}];')

    return lines


def generate_dot_output(
    analyzer: CallGraphAnalyzer,
    level: str = 'medium',
    module_filter: Optional[str] = None,
) -> str:
    """Generate Graphviz dot output.

    Args:
        analyzer: Analyzed call graph
        level: Detail level
            - low:    Key functions story-driven (phases + their immediate calls)
            - medium: Public APIs + phases (broader view)
            - high:   All functions with signatures + tooltips (complete, default)
        module_filter: Only show specific module (e.g., 'foo.bar')

    Returns:
        Dot language string
    """
    lines = []

    # Graph header (styled like bsdconfig)
    lines.append('digraph bpydot {')
    lines.append(f"    // Generated by bpydot {VERSION}")
    lines.append('    // Inspired by FreeBSD bsdconfig API module')
    lines.append('')
    lines.append('    // Graph styling (bsdconfig layout)')
    lines.append('    graph [')
    lines.append('        labelloc=top,         // display label at top of graph')
    lines.append('        rankdir=LR,           // create ranks left-to-right (vertical format)')
    lines.append('        orientation=portrait, // default')
    lines.append('        ratio=fill,           // approximate aspect ratio')
    lines.append('        center=1,             // center drawing on page')
    lines.append('        compount=true,        // allow edges between clusters')
    lines.append('        clusterrank=local,    // keep cluster nodes together')
    lines.append('        concentrate=true,     // edge concentrators (may crash old Graphviz)')

    # Spline type varies by level for optimal readability
    if level == 'low':
        lines.append('        splines=curved,       // curved edges for sparse low output')
    else:
        lines.append('        splines=polyline,     // polyline edges for denser output')

    lines.append('        ranksep=0.75,')
    lines.append('        nodesep=0.5,')
    lines.append('        bgcolor=white,')
    lines.append('        fontname="Helvetica-Bold",')
    lines.append('        fontsize=18')
    lines.append('    ];')
    lines.append('')

    # Build command line representation for graph title
    cmd_parts = ['bpydot']
    cmd_parts.append(f"-l {level}")
    if module_filter:
        cmd_parts.append(f"-m {module_filter}")
    cmd_line = ' '.join(cmd_parts)

    # Graph title (OUTSIDE bounding box, bsdconfig style)
    lines.append('    label="bpydot ' + VERSION_VERBOSE + '\\n' +
                 cmd_line + '\\nGenerated: ' +
                 __import__('datetime').datetime.now().strftime('%F %H:%M') + '";')
    lines.append('')

    # Wrap entire graph in a subgraph for bounding box (bsdconfig technique)
    lines.append('    // Main bounding box (for printing)')
    lines.append('    subgraph cluster_main {')
    lines.append('        style=solid;')
    lines.append('        penwidth=2.0;')
    lines.append('        color="#333333";')
    lines.append('        label="";  // No label on cluster (title is at graph level)')
    lines.append('        margin=20; // Add margin/padding inside bounding box (bsdconfig style)')
    lines.append('')
    lines.append('    // Node defaults')
    lines.append('    node [')
    lines.append('        fontname="Helvetica",')
    lines.append('        fontsize=11,')
    lines.append('        style=filled,')
    lines.append('        fillcolor=white,')
    lines.append('        penwidth=1.5,')
    lines.append('        margin="0.15,0.10"')
    lines.append('    ];')
    lines.append('')
    lines.append('    // Edge defaults')
    lines.append('    edge [')
    lines.append('        fontname="Helvetica",')
    lines.append('        fontsize=9,')
    lines.append('        arrowsize=0.7,')
    lines.append('        color="#424242"')
    lines.append('    ];')
    lines.append('')

    # Group by subsystem using NESTED subgraphs (show file structure)
    # Build hierarchy: subsystem -> [path, path, ...] -> [modules]
    # We need to build a tree structure for multi-level nesting
    hierarchy = defaultdict(lambda: defaultdict(list))
    for package, module in analyzer.modules.items():
        if module_filter and not package.startswith(module_filter):
            continue

        # package format examples:
        # - "cli" -> top-level file cli.py
        # - "foo.bar.baz" -> foo/bar/baz.py
        parts = package.split('.')

        if len(parts) == 1:
            # Top-level module: use "." for root directory (Unix convention)
            subsystem = "."
            file_path = parts[0]
        else:
            # Nested module: first part is subsystem, rest is file path
            subsystem = parts[0]
            file_path = '/'.join(parts[1:])  # Use / to indicate directory structure

        hierarchy[subsystem][file_path].append(module)

    # NB: First, determine which nodes appear in edges (must be defined)
    # NB: Must respect level filtering - only track nodes from edges that will be rendered
    nodes_in_edges = set()

    # Build node-to-subsystem map first (needed for filtering)
    temp_node_to_subsystem = {}
    for package, module in analyzer.modules.items():
        parts = package.split('.')
        if len(parts) == 1:
            subsystem = '.'
        else:
            subsystem = parts[0]
        for func_name in module.functions.keys():
            node_id = f"{package}_{func_name}".replace('.', '_')
            temp_node_to_subsystem[node_id] = subsystem

    for caller_full, callee_full in analyzer.call_edges:
        # Apply same filters as edge rendering will apply
        caller_id = caller_full.replace('.', '_')
        callee_id = callee_full.replace('.', '_')
        caller_subsystem = temp_node_to_subsystem.get(caller_id)
        callee_subsystem = temp_node_to_subsystem.get(callee_id)

        # Extract function names
        caller_func_name = caller_full.split('.')[-1]
        callee_func_name = callee_full.split('.')[-1]

        # LOW LEVEL: Apply same filtering as edge rendering
        if level == 'low':
            # Skip inter-subgraph edges
            if caller_subsystem != callee_subsystem:
                continue

            # Skip edges to/from private functions
            if caller_func_name.startswith('_') or callee_func_name.startswith('_'):
                continue

            # Skip edges to/from introspection subsystem
            if caller_subsystem == 'introspection' or callee_subsystem == 'introspection':
                continue

        # MEDIUM LEVEL: Apply same filtering as edge rendering
        if level == 'medium':
            # Skip edges to/from introspection subsystem
            if caller_subsystem == 'introspection' or callee_subsystem == 'introspection':
                continue

        # Only add nodes if this edge will actually be rendered
        nodes_in_edges.add(caller_full)
        nodes_in_edges.add(callee_full)

    # Generated NESTED subgraphs for each subsystem
    all_phase_nodes = []
    all_public_nodes = []
    all_internal_nodes = []

    for subsystem in sorted(hierarchy.keys()):
        files_in_subsystem = hierarchy[subsystem]
        color = MODULE_COLORS.get(subsystem, '#E0E0E0')

        # SIMPLIFICATION: If subsystem has only one file, flatten the label
        if len(files_in_subsystem) == 1:
            single_file = list(files_in_subsystem.keys())[0]
            cluster_label = f"{subsystem}/{single_file}.py"  # Add .py extension for clarity
            # Don't create nested subgraph for single-file subsystems
            # Escape subsystem name for cluster ID (e.g., "." becomes "root")
            cluster_id = subsystem.replace('.', 'root')
            lines.append(f"    subgraph cluster_{cluster_id} {{")
            lines.append(f'        label="{cluster_label}";')
            lines.append('        style="rounded,filled";')
            lines.append(f'        fillcolor="{color}";')
            lines.append('        penwidth=2.0;')
            lines.append('        fontsize=14;')
            lines.append('        fontname="Helvetica-Bold";')
            lines.append('')

            # Add function nodes directly (no nested subgraph)
            for module in files_in_subsystem[single_file]:
                for func_name, func in module.functions.items():
                    full_name = f"{module.package}.{func_name}"

                    # Level filtering using helper function
                    if not should_show_node(func, full_name, level, subsystem, nodes_in_edges):
                        continue

                    # Track node by category
                    node_id = f"{module.package}_{func_name}".replace('.', '_')
                    _, _, _, _, node_category = get_node_color_and_style(func)

                    if node_category == 'phase':
                        all_phase_nodes.append(node_id)
                    elif node_category == 'public':
                        all_public_nodes.append(node_id)
                    else:
                        all_internal_nodes.append(node_id)

                    # Generate node definition
                    lines.extend(generate_function_node_lines(func, module, level))

            # Add global variable nodes for single-file subsystems (only at high level)
            if level == 'high':
                for module in files_in_subsystem[single_file]:
                    for global_name in module.globals:
                        # FILTER: Skip if this global is imported from another module
                        if hasattr(module, 'imported_vars') and global_name in module.imported_vars:
                            continue

                        lines.extend(generate_global_node_lines(global_name, module, analyzer))

            lines.append('    }')
            lines.append('')
            continue

        # MULTI-FILE SUBSYSTEM: Create nested structure
        # Escape subsystem name for cluster ID
        cluster_id = subsystem.replace('.', 'root')
        lines.append(f"    subgraph cluster_{cluster_id} {{")
        lines.append(f'        label="{subsystem}";')  # Keep original case for copy/paste accuracy
        lines.append('        style="rounded,filled";')
        lines.append(f'        fillcolor="{color}";')
        lines.append('        penwidth=2.0;')
        lines.append('        fontsize=14;')
        lines.append('        fontname="Helvetica-Bold";')
        lines.append('')

        # Create nested subgraphs for each file in this subsystem
        for file_path in sorted(files_in_subsystem.keys()):
            modules_list = files_in_subsystem[file_path]
            # Use the file_path directly as the label (it now has / separators)
            cluster_id = f"{subsystem}_{file_path}".replace('.', '_').replace('/', '_')

            lines.append(f"        subgraph cluster_{cluster_id} {{")
            # Full path from project root
            lines.append(f'            label="{subsystem}/{file_path}.py";')
            lines.append('            style="rounded";')
            lines.append('            penwidth=1.0;')
            lines.append('            fontsize=10;')
            lines.append('')

            # Add function nodes from this file
            for module in modules_list:
                for func_name, func in module.functions.items():
                    full_name = f"{module.package}.{func_name}"

                    # Level filtering using helper function
                    if not should_show_node(func, full_name, level, subsystem, nodes_in_edges):
                        continue

                    # Track node by category
                    node_id = f"{module.package}_{func_name}".replace('.', '_')
                    _, _, _, _, node_category = get_node_color_and_style(func)

                    if node_category == 'phase':
                        all_phase_nodes.append(node_id)
                    elif node_category == 'public':
                        all_public_nodes.append(node_id)
                    else:
                        all_internal_nodes.append(node_id)

                    # Generate node definition
                    lines.extend(generate_function_node_lines(
                        func, module, level, indent='            '))

            # Add global variable nodes (only at high level)
            if level == 'high':
                for module in modules_list:
                    # Show ALL module-level globals
                    # This reveals vestigial/unused globals - important for technical debt!
                    for global_name in module.globals:
                        # FILTER: Skip if this global is imported from another module
                        # NB: it's defined elsewhere, not here
                        if hasattr(module, 'imported_vars') and global_name in module.imported_vars:
                            continue  # Don't create node - it belongs to the defining module

                        lines.extend(generate_global_node_lines(global_name, module, analyzer,
                                                                indent='            '))

            # Close file cluster
            lines.append('        }')
            lines.append('')

        lines.append('    }')
        lines.append('')

    # NOTE: rank=same constraints BREAK cluster containment in Graphviz!
    # They pull nodes out of their subgraphs, causing the "floater" effect.
    # Removed these constraints to preserve cluster membership.
    # The natural LR rankdir provides sufficient layering without explicit ranks.

    # ORPHAN NODES: Define any nodes that appear in edges but weren't in clusters.
    # This catches nodes that were filtered out or otherwise missed.
    defined_nodes_set = set()
    for package, module in analyzer.modules.items():
        for func_name in module.functions.keys():
            full_name = f"{module.package}.{func_name}"
            defined_nodes_set.add(full_name)

    orphan_nodes = []
    for full_name in nodes_in_edges:
        if full_name not in defined_nodes_set:
            continue  # Node doesn't exist in analyzer

        # Check if this node appears in the hierarchy output
        # If it's in nodes_in_edges but we haven't output it yet, it's an orphan
        func = analyzer.functions.get(full_name)
        if not func:
            continue

        # Check if it was filtered out (private functions, etc.)
        # We need to track which nodes we actually output above
        # For now, assume any private functin is an orphan candidate
        is_private = func.name.startswith('_') and not (func.name.startswith('__') and
                                                        func.name.endswith('__'))

        # Also check against our level filter to see if it would have been included
        subsystem = full_name.split('.', maxsplit=1)[0] if '.' in full_name else 'root'
        show_node = should_show_node(func, full_name, level, subsystem, nodes_in_edges)

        # If it's in edges but wouldn't be shown by level filter, it's an orphan
        if not show_node or is_private:
            orphan_nodes.append((full_name, func, is_private))

    if orphan_nodes:
        lines.append('')
        lines.append('    // Orphan nodes (referenced but not in main hierarchy)')
        for full_name, func, is_private in orphan_nodes:
            node_id = full_name.replace('.', '_')
            label = func.name

            if is_private:
                # Private functions: dashed border, gray, to indicate "internal implementation"
                lines.append(f"    {node_id} [")
                lines.append(f'        label="{label}",')
                lines.append('        fillcolor="#F5F5F5",')
                lines.append('        style="filled,dashed",')
                lines.append('        shape="ellipse",')
                lines.append('        penwidth=1.0,')
                lines.append('        color="#999999",')
                lines.append('        fontcolor="#666666"')
                lines.append('      ];')
            else:
                # Other orphans: light gray but solid border
                lines.append(f"    {node_id} [")
                lines.append(f'        label="{label}",')
                lines.append('        fillcolor="#FAFAFA",')
                lines.append('        style="filled",')
                lines.append('        shape="ellipse",')
                lines.append('        penwidth=0.8,')
                lines.append('        color="#AAAAAA",')
                lines.append('        fontcolor="#777777"')
                lines.append('      ];')

        lines.append('')

    # Add call edges
    lines.append('    // Call edges')

    # Build a map of node_id -> subsystem for edge styling
    node_to_subsystem = {}
    for package, module in analyzer.modules.items():
        # Determine subsystem the same way we did for hierarchy
        parts = package.split('.')
        if len(parts) == 1:
            subsystem = '.'
        else:
            subsystem = parts[0]

        for func_name in module.functions.keys():
            node_id = f"{package}_{func_name}".replace('.', '_')
            node_to_subsystem[node_id] = subsystem

    for caller_full, callee_full in analyzer.call_edges:
        # Filter by module if specified
        if module_filter:
            if not caller_full.startswith(module_filter) and not \
                callee_full.startswith(module_filter):
                continue

        # Convert to node IDs
        caller_id = caller_full.replace('.', '_')
        callee_id = callee_full.replace('.', '_')

        # Get subsystems for caller and callee
        caller_subsystem = node_to_subsystem.get(caller_id)
        callee_subsystem = node_to_subsystem.get(callee_id)

        # Extract function names for filtering
        caller_func_name = caller_full.split('.')[-1]
        callee_func_name = callee_full.split('.')[-1]

        # LOW LEVEL: Additional filtering to prevent auto-reated nodes
        if level == 'low':
            # Skip inter-subgraph edges
            if caller_subsystem != callee_subsystem:
                continue

            # Skip edges to/from private functions (starts with _)
            if caller_func_name.startswith('_') or callee_func_name.startswith('_'):
                continue

            # Skip edges to/from introspection subsystem (filtered at low level)
            if caller_subsystem == 'introspection' or callee_subsystem == 'introspection':
                continue

        # MEDIUM LEVEL: Filter introspection subsystem only
        if level == 'medium':
            # Skip edges to/from introspection subsystem
            if caller_subsystem == 'introspection' or callee_subsystem == 'introspection':
                continue

        # Determine edge style and weight
        style = EDGE_STYLES['direct']

        # CRITICAL: Add weight to keep intra-cluster edges tight (bsdconfig technique)
        weight = '20' if caller_subsystem == callee_subsystem else '1'

        # Color cross-cluster edges by source cluster color
        if caller_subsystem and callee_subsystem and caller_subsystem != callee_subsystem:
            # Cross-cluster edge - use darker source cluster color (visible on white)
            source_color = EDGE_COLORS.get(caller_subsystem, '#424242')
            lines.append(f"    {caller_id} -> {callee_id} [")
            lines.append(f'        style="{style["style"]}",')
            lines.append(f'        penwidth="{style["penwidth"]}",')
            lines.append(f'        color="{source_color}",')
            lines.append(f'        weight="{weight}"')
            lines.append('    ];')
        else:
            # Intra-cluster edge - use default color
            lines.append(f"    {caller_id} -> {callee_id} [")
            lines.append(f'        style="{style["style"]}",')
            lines.append(f'        penwidth="{style["penwidth"]}",')
            lines.append(f'        color="{style["color"]}",')
            lines.append(f'        weight="{weight}"')
            lines.append('    ];')

    # GLOBAL-TO-FUNCTION EDGES (only at high level)
    # Show where globals are imported and used (CONSUMER edges)
    if level == 'high':
        lines.append('')
        lines.append('    // Global variable usage edges (consumers)')

        # Build reverse mapping: var_name -> defining_package
        # This works for ALL globals, not just those in variable_mappings
        global_definitions = {}  # var_name -> package whre it's defined
        for pkg, mod in analyzer.modules.items():
            for global_name in mod.globals:
                # Only track if not imported (i.e., defined here)
                if not (hasattr(mod, 'imported_vars') and global_name in mod.imported_vars):
                    global_definitions[global_name] = pkg

        # Now create edges from globals to functions that use them
        # SIMPLIFIED APPROACH: Check ALL functions in ALL modules for usage of each global
        # This catches both module-level imports AND in-function imports
        for var_name, defining_package in global_definitions.items():
            global_id = f"{defining_package}_{var_name}".replace('.', '_')

            # Check every module and every function for usage of this global
            for package, module in analyzer.modules.items():
                if hasattr(module, 'source_code'):
                    for func_name, func_obj in module.functions.items():
                        # Get function source with SMART boundaries
                        source_lines = module.source_code.split('\n')
                        if hasattr(func_obj, 'lineno') and hasattr(func_obj, 'end_lineno') and \
                            func_obj.end_lineno:
                            # Precise boundaries available
                            func_start = func_obj.lineno - 1
                            func_end = func_obj.end_lineno
                            func_source = '\n'.join(source_lines[func_start:func_end])
                        elif hasattr(func_obj, 'lineno'):
                            # No end_lineno - use smart heuristic: find next top-level def/class
                            func_start = func_obj.lineno - 1
                            # Find next function or class at same indentation level
                            func_end = len(source_lines)  # Default to end of file
                            for i in range(func_start + 1, len(source_lines)):
                                line = source_lines[i]
                                # Check for next top-level def (no leading space before def/class)
                                if line.startswith('def ') or line.startswith('class ') or \
                                    line.startswith('########'):
                                    func_end = i
                                    break
                            func_source = '\n'.join(source_lines[func_start:func_end])
                        else:
                            continue

                        # Check if this function uses the global variable
                        pattern = r'\b' + re.escape(var_name) + r'\b'
                        if re.search(pattern, func_source):
                            # Check it's not just in strings (must be actual code usage)
                            lines_with_var = [line for line in func_source.split('\n')
                                if re.search(pattern, line)]
                            has_real_usage = False
                            for line in lines_with_var:
                                # Remove comments first
                                code_part = line.split('#')[0] if '#' in line else line

                                # Skip if variable is ONLY inside string literals. Strategy: Remove
                                # all string literals, then check if variable still appears. This
                                # correctly handles: "text" + VAR + "text" (VAR remains after string
                                # removal) and filters: "text with VAR inside" (VAR disappears after
                                # string removal)

                                # Remove string literals (simple approach: remove "..." and '...')
                                code_without_strings = re.sub(r'"[^"]*"', '""', code_part)
                                code_without_strings = re.sub(r"'[^']*'", "''",
                                    code_without_strings)

                                # Check if variable still appears after string removal
                                if re.search(pattern, code_without_strings):
                                    has_real_usage = True
                                    break

                            if has_real_usage:
                                consumer_id = f"{package}_{func_name}".replace('.', '_')
                                # Edge from global to function (dotted purple to show data flow)
                                lines.append(f"    {global_id} -> {consumer_id} [")
                                lines.append('        style="dotted",')
                                lines.append('        penwidth=1.5,')
                                lines.append('        color="#9C27B0",')
                                lines.append('        weight=5')
                                lines.append('    ];')

        # FUNCTION-TO-GLOBAL EDGES (PRODUCER edges)
        # Show where globals are assigned from function results
        # e.g., FOO = bar()
        lines.append('')
        lines.append('    // Global variable producer edges (function → global)')

        for var_name, func_full_name in analyzer.variable_mappings.items():
            # FILTER: Skip if the "function" is actually an external import (e.g., threading.Lock)
            # Only create edges for functions that actually exist in our codebase
            if func_full_name not in analyzer.functions:
                continue  # External import, not a real producer function

            # Find the package where this global is defined
            if var_name not in global_definitions:
                continue  # Global not tracked

            defining_package = global_definitions[var_name]
            global_id = f"{defining_package}_{var_name}".replace('.', '_')

            # The function might be in the same module or different
            # func_full_name format: "package.function_name"
            producer_id = func_full_name.replace('.', '_')

            # Edge from function to global (solid purple to show creation/assignment)
            # Direction: bar() → FOO
            lines.append(f"    {producer_id} -> {global_id} [")
            lines.append('        style="solid",')
            lines.append('        penwidth=2.0,')
            lines.append('        color="#7B1FA2",')
            lines.append('        weight=10')
            lines.append('    ];')

    # SYNTHETIC EDGES: Add dotted edges for isolated nodes
    # Collect all node IDs that were actually defined
    defined_nodes = set()
    for package, module in analyzer.modules.items():
        if module_filter and not package.startswith(module_filter):
            continue
        for func_name, func in module.functions.items():
            full_name = f"{module.package}.{func_name}"
            # Must match the show_node logic from above
            subsystem_name = full_name.split('.', maxsplit=1)[0] if '.' in full_name else 'root'
            show_node = should_show_node(func, full_name, level, subsystem_name, nodes_in_edges)

            if show_node:
                node_id = f"{module.package}_{func_name}".replace('.', '_')
                defined_nodes.add(node_id)

    # Find isolated nodes (defined but not in edges)
    nodes_with_edges = set()
    for caller_full, callee_full in analyzer.call_edges:
        caller_id = caller_full.replace('.', '_')
        callee_id = callee_full.replace('.', '_')
        if caller_id in defined_nodes:
            nodes_with_edges.add(caller_id)
        if callee_id in defined_nodes:
            nodes_with_edges.add(callee_id)

    isolated_nodes = defined_nodes - nodes_with_edges

    # LOW LEVEL: Don't add synthetic edges - they ruin the clean layout
    # Let isolated nodes remain isolated for visual clarity
    if isolated_nodes and level != 'low':
        lines.append('')
        lines.append('    // Synthetic edges for isolated nodes (implicit relationships)')

        # Strategy: Connect isolated utility functions to their subsystem's "main" function or to a
        # related function based on naming patterns
        for isolated_id in sorted(isolated_nodes):
            # Extract subsystem from node_id
            subsystem = node_to_subsystem.get(isolated_id)

            # Find a related function in the same subsystem that HAS edges
            related_candidates = [
                node_id for node_id in nodes_with_edges
                if node_to_subsystem.get(node_id) == subsystem
            ]

            if related_candidates:
                # Connect to the first candidate (alphabetically sorted for stability)
                target = sorted(related_candidates)[0]
                lines.append(f"    {isolated_id} -> {target} [")
                lines.append('        style="dashed",')
                lines.append('        penwidth=1.0,')
                lines.append('        color="#9E9E9E",')
                lines.append('        arrowhead="odot",')
                lines.append('        weight=1,')
                lines.append('        constraint=false')
                lines.append('    ];')

    # Close main bounding box
    lines.append('    }')
    lines.append('}')

    return '\n'.join(lines)


def run_api_introspection(
    root_path: str,
    level: str = 'medium',
    module_filter: Optional[str] = None,
    output_file: Optional[str] = None,
):
    """Main entry point for API introspection.

    Args:
        root_path: Path to package root directory
        level: Detail level (minimal, low, medium, high, full)
        module_filter: Only analyze specific module
        output_file: Write output to file (or stdout if None)
    """
    print(f"Analyzing codebase: {root_path}", file=sys.stderr)

    # Analyze code
    analyzer = CallGraphAnalyzer(root_path)
    analyzer.analyze()

    m = len(analyzer.modules)
    f = len(analyzer.functions)
    print(f"Found {m} {plur(m, 'module')}, {f} {plur(f, 'function')}", file=sys.stderr)

    # Generate dot output
    output = generate_dot_output(analyzer, level, module_filter)

    # Write output
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Wrote API visualization to: {output_file}", file=sys.stderr)
    else:
        print(output)


################################################################################
# END
################################################################################
