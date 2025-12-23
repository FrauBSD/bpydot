# bpydot

**Python code analysis and visualization tool**

Inspired by FreeBSD `bsdconfig`'s API module, `bpydot` provides award-winning call graph generation for Python codebases using AST-based introspection.

## Features

- **AST-Based Analysis**: Deep static code introspection
- **Call Graph Generation**: Graphviz dot format output
- **Global Variable Tracking**: Producer/consumer edge visualization
- **Multiple Detail Levels**: Low (story-driven), Medium (APIs), High (complete)
- **Visual Clustering**: Subgraph grouping by module/subsystem
- **Technical Debt Detection**: Identifies unused globals, vestigial code, duplicates

## Installation

```bash
# From source
cd bpydot
pip install -e .

# Or directly
python3 -m pip install .
```

## Usage

```bash
# Analyze current directory (high detail, stdout)
bpydot

# Analyze specific project
bpydot -l medium -o api.dot /path/to/project

# Story-driven overview (low detail)
bpydot -l low -o overview.dot

# Generate SVG from dot output
bpydot -o api.dot && dot -Tsvg api.dot -o api.svg
```

## Detail Levels

### Low (Story-Driven)
- Key functions only
- Intra-module edges only
- Clean, digestible overview
- Perfect for understanding program flow

### Medium
- Public API + phase orchestrators
- All edges (including cross-module)
- Function signatures
- Docstring tooltips

### High (Complete)
- All functions (public + private)
- All global variables (hexagon nodes)
- Producer edges (function -> global, solid purple)
- Consumer edges (global -> function, dotted purple)
- Full signatures with return types
- Docstring tooltips

## Use Cases

### Pre-Flight PR Review
```bash
bpydot -o before.dot
# Make changes
bpydot -o after.dot
# Compare to identify new technical debt
```

### Architecture Exploration
```bash
# Get the big picture
bpydot -l low -o overview.dot && dot -Tsvg overview.dot -o overview.svg

# Dive deeper
bpydot -l high -o complete.dot && dot -Tsvg complete.dot -o complete.svg
```

### Technical Debt Analysis
High deteail level reveals:
- Unused global variables (no consumer edges)
- Vestigial constants (duplicates across modules)
- Orphaned functions (no callers)
- Duplicate implementations

## Output Format

Generates Graphviz dot format with:
- **Node Shapes**:
  - Rectangle: Regular functions
  - Hexagon: Global variables (purple fill)
  - Diamond: Decision functions (returns bool)
  - Dashed border: Private functions (`_name`)

- **Edge styles**:
  - Solid black: Function calls
  - Solid purple: Producer edges (function -> global)
  - Dotted purple: Consumer edges (global -> function)
  - Color-coded: Cross-module edges use source module color

- **Clustering**: Automatic subgraph grouping by module hierarchy

## Example Output

```bash
bpydot ~/foo -o foo.dot
# Wrote API visualization to: foo.dot
# Generated high-detail call graph:
#   43 modules
#   287 functions
#   43 globals
#   891 edges

dot -Tsvg foo.dot -o foo.svg
```

## Requirements

- Python 3.6+
- Graphviz (for rendering SVG/PNG from dot files)

## Design Philosophy

Unlike generic code analyzers that produce cluttered, unusable graphs, `bpydot` is designed with UX in mind:

1. **Vertical Document Flow**: Uses `rankdir=LR` to create scrollable documents
2. **Visual Hierarchy**: Color-coded modules, shape-coded node types
3. **Filtered Noise**: Low/medium levels hide implementation details
4. **Story-Driven**: Low level shows only essential program flow
5. **Technical Depth**: High level reveals everything for deep analysis

## Credits

Inspired by and following the design principles of:
- **FreeBSD `bsdconfig` API module** (Devin Teske)
  - Award-winning call graph generation
  - Story-driven visualizations
  - Practical pre-flight analysis

## License

BSD 2-Clause

## Author

**Devin Teske** <dteske@FreeBSD.org>
