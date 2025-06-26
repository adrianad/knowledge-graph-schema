# Relational Knowledge Graph

A Python application that extracts database schemas and creates knowledge graphs to help LLMs understand table relationships for more efficient SQL generation.

## Features

- **Schema Extraction**: Connects to SQLite, PostgreSQL, and MySQL databases
- **Knowledge Graph**: Creates NetworkX-based graphs showing table relationships
- **Smart Analysis**: Finds relevant tables based on keywords and query intent
- **Visualization**: Interactive Plotly graphs and static matplotlib plots
- **Community Detection**: Identifies clusters of related tables
- **CLI Interface**: Easy-to-use command line tools

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Analyze your database schema:**
```bash
rkg analyze -c "sqlite:///your_database.db"
```

2. **Find relevant tables for your query:**
```bash
rkg find-tables -c "sqlite:///your_database.db" -k "user,order,product"
```

3. **Visualize the schema graph:**
```bash
rkg visualize -c "sqlite:///your_database.db" -o schema_graph.html
```

## Usage Examples

### Analyze Database Schema
```bash
# Basic analysis
rkg analyze -c "postgresql://user:pass@localhost/db"

# Save graph data
rkg analyze -c "sqlite:///ecommerce.db" -o graph_data.json
```

### Find Relevant Tables
```bash
# Find tables related to specific keywords
rkg find-tables -c "sqlite:///shop.db" -k "customer,order" -m 5

# Include related tables in results
rkg find-tables -c "sqlite:///shop.db" -k "product" -r
```

### Get Join Suggestions
```bash
# Suggest tables to join with existing ones
rkg suggest-joins -c "sqlite:///shop.db" -t "orders,customers"
```

### Generate Visualizations
```bash
# Interactive HTML visualization
rkg visualize -c "sqlite:///shop.db" -o interactive_graph.html

# Different layout algorithms
rkg visualize -c "sqlite:///shop.db" -l hierarchical
```

### Schema Summary
```bash
# Get comprehensive schema statistics
rkg summary -c "sqlite:///shop.db"
```

## Connection Strings

### SQLite
```
sqlite:///path/to/database.db
```

### PostgreSQL
```
postgresql://username:password@host:port/database
```

### MySQL
```
mysql+pymysql://username:password@host:port/database
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env with your database connection details
```

## Architecture

The application is built with a modular architecture supporting future Neo4j integration:

- **DatabaseExtractor**: Handles schema extraction from various databases
- **SchemaGraph**: Creates and manages NetworkX knowledge graphs
- **SchemaAnalyzer**: Provides intelligent table discovery and analysis
- **GraphVisualizer**: Creates static and interactive visualizations
- **CLI**: Command-line interface for all operations

## Use Cases

### For LLM SQL Generation
1. Extract schema and build knowledge graph
2. Use keywords to find relevant tables
3. Export focused schema subset
4. Provide subset to LLM for accurate SQL generation

### Schema Understanding
- Visualize complex database relationships
- Identify table clusters and communities
- Find connection paths between tables
- Analyze schema structure and importance

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
```

## Future Enhancements

- Neo4j backend support (architecture ready)
- LLM integration for semantic table mapping
- Advanced relationship scoring
- Schema evolution tracking
- Web API interface