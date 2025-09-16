# Relational Knowledge Graph

A Python application that extracts database schemas and creates knowledge graphs to help LLMs understand table relationships for more efficient SQL generation.

## Features

- **Schema Extraction**: Connects to SQLite, PostgreSQL, and MySQL databases
- **Dual Backend Support**: NetworkX for development, Neo4j for production scale
- **Knowledge Graph**: Creates sophisticated graphs with table importance scoring
- **Advanced Clustering**: 
  - Community detection using Louvain algorithm
  - Importance-based clustering around hub tables
  - Overlapping clusters for realistic business modeling
- **LLM Integration**: Automatic cluster naming, keyword extraction, and semantic analysis
- **Path Discovery**: Find all connection paths between specific tables for SQL JOIN planning
- **Smart Analysis**: Finds relevant tables based on keywords and natural language queries
- **Visualization**: Interactive Plotly graphs and static matplotlib plots
- **CLI Interface**: Comprehensive command line tools for all operations

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

2. **Create table clusters:**
```bash
rkg create-clusters -c "sqlite:///your_database.db"
```

3. **Get main cluster (most important tables):**
```bash
rkg get-main-cluster
```

4. **List additional clusters:**
```bash
rkg list-clusters
```

5. **Explore specific cluster:**
```bash
rkg show-cluster -i cluster_3
```

## Usage Examples

### Analyze Database Schema
```bash
# Basic analysis
rkg analyze -c "postgresql://user:pass@localhost/db"

# Save graph data
rkg analyze -c "sqlite:///ecommerce.db" -o graph_data.json
```

### Work with Table Clusters
```bash
# Create clusters with importance-based method
rkg create-clusters -c "sqlite:///shop.db" --method importance

# Get the most important tables
rkg get-main-cluster --detailed

# List remaining clusters
rkg list-clusters --exclude-main
```

### Find Connection Paths Between Tables
```bash
# Find all connections between specific tables (most useful for LLM query planning)
rkg find-path -t "users,orders,products" --max-hops 3

# Get complete connection map for a set of related tables
rkg find-path -t "customers,invoices,payments,products" --max-hops 2
```

### Get Join Suggestions  
```bash
# Suggest tables to join with existing ones (traditional approach)
rkg suggest-joins -c "sqlite:///shop.db" -t "orders,customers" --max-hops 2

# Get suggestions organized per base table
rkg suggest-joins -c "sqlite:///shop.db" -t "orders,customers" --per-table
```

### Generate Visualizations
```bash
# Interactive HTML visualization
rkg visualize -c "sqlite:///shop.db" -o interactive_graph.html

# Different layout algorithms
rkg visualize -c "sqlite:///shop.db" -l hierarchical
```

### Advanced Clustering
```bash
# Create clusters using community detection
rkg create-clusters -c "sqlite:///shop.db" --method community

# Create importance-based clusters with custom parameters
rkg create-clusters -c "sqlite:///shop.db" --method importance \
  --min-size 3 --max-hops 2 --top-pct 0.25

# Requires Neo4j backend for storage
rkg create-clusters -c "sqlite:///shop.db" --backend neo4j
```

### LLM-Powered Analysis
```bash
# Extract business keywords using LLM
rkg llm-keyword-extraction -c "sqlite:///shop.db"

# Create clusters with LLM analysis
rkg create-clusters -c "sqlite:///shop.db" --llm

# Explore table relationships
rkg explore-table -c "sqlite:///shop.db" -t "orders" --hops 3
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

Create a `.env` file for configuration:

```bash
# Database connection (optional - can use -c flag instead)
DATABASE_URL=sqlite:///your_database.db

# Neo4j configuration (for advanced features)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM configuration (for semantic analysis)
OPENAI_API_BASE=http://localhost:8000/v1  # Local LLM server
OPENAI_API_KEY=dummy-key
LLM_MODEL=qwen
LLM_MAX_CONCURRENT=5

# SQL execution timeout (default: 5000ms)
SQL_STATEMENT_TIMEOUT_MS=5000
```

## Architecture

The application is built with a modular, extensible architecture:

- **DatabaseExtractor**: Handles schema extraction from SQLite, PostgreSQL, MySQL
- **Dual Backend System**:
  - **NetworkXBackend**: Fast, in-memory graphs for development and smaller schemas
  - **Neo4jBackend**: Persistent, scalable graphs for production and large schemas
- **SchemaGraph**: Creates knowledge graphs with advanced relationship analysis
- **SchemaAnalyzer**: Intelligent table discovery with importance scoring
- **LLMClusterAnalyzer**: AI-powered cluster naming and keyword extraction
- **GraphVisualizer**: Interactive and static visualizations
- **CLI**: Comprehensive command-line interface

## Clustering Algorithms

### Community Detection
- **Algorithm**: Louvain method for community detection
- **Use Case**: Discover organic clusters based on relationship density
- **Output**: Non-overlapping clusters of naturally related tables

### Importance-Based Clustering
- **Algorithm**: Hub-and-spoke clustering around high-importance tables  
- **Importance Scoring**: PageRank (70%) + Degree Centrality (30%)
- **Features**: 
  - Overlapping clusters (tables can belong to multiple clusters)
  - Configurable cluster size, relationship distance, and core selection
  - Better representation of real business domains

## Use Cases

### For LLM SQL Generation
1. Extract schema and build knowledge graph
2. Get main cluster or find relevant table clusters  
3. Use `find-path` to discover JOIN patterns between target tables
4. Export focused schema subset with connection paths
5. Provide subset to LLM for accurate SQL generation

### Database Query Planning
- **find-path**: Get complete connection map for specific tables (ideal for JOIN construction)
- **suggest-joins**: Discover additional tables based on importance (good for exploration)
- **get-main-cluster**: Start with most important tables for LLM context optimization

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

## MCP Tools for LLM Integration

The application provides a comprehensive Model Context Protocol (MCP) server with 8 specialized tools for autonomous database exploration and query construction. Use the following description in your LLM system prompt:

### LLM System Prompt - Database Tools

#### Simplified Version
```
You have 8 MCP tools for database exploration:

**Schema Tools:**
- `explore_table("table1,table2")` - Get table schemas  
- `explore_view("view1,view2")` - Get view schemas
- `get_main_cluster()` - Get most important tables

**Domain Tools:**
- `list_clusters()` - List business domain clusters
- `show_cluster("cluster_id")` - Get tables from specific domain

**Relationship Tools:**
- `find_path("table1,table2,table3")` - Find JOIN paths between tables (primary tool for SQL construction)
- `suggest_joins("base_table")` - Discover additional relevant tables  
- `find_related_views("table1,table2")` - Find statistical views

**Workflow:** Start with `get_main_cluster()` → Use `find_path()` for JOINs → Use `explore_table()` for schemas
```

#### Detailed Version
```
You have access to 8 database exploration tools through MCP. Use these tools to understand database schemas and construct efficient SQL queries:

## Schema Discovery Tools

**explore_table(table_names)**
- Get detailed information about specific tables from Neo4j graph
- Input: "user_,booking,sample" (comma-separated table names to explore)
- Returns: DDL and column details for specified tables
- Use when: You need schema details for known tables

**explore_view(view_names)**
- Get detailed information about specific database views for statistics and reporting
- Input: "sales_summary,monthly_stats" (comma-separated view names to explore)
- Returns: View schemas with clear (VIEW) labels
- Use when: You need schema details for statistical/reporting views

**get_main_cluster(detailed=False)**  
- Get the main cluster (union of top N most important clusters) without duplicates
- detailed: Whether to include detailed DDL information (default: False)
- Returns: Clean list of essential tables (no metadata noise)
- Use when: Starting exploration or need core tables for general queries

## Business Domain Tools

**list_clusters(exclude_main=True)**
- List all available table clusters from Neo4j
- exclude_main: Whether to exclude the clusters that make up the main cluster (default: True)
- Returns: Cluster names, descriptions, keywords, table counts (filtered to exclude main cluster tables)
- Use when: You want to explore business domains or find related table groups

**show_cluster(cluster_id, detailed=False, exclude_main=True)**
- Show detailed information about a specific cluster
- cluster_id: The cluster ID to show details for
- detailed: Whether to include detailed column information for tables (default: False)
- exclude_main: Whether to exclude tables that are in the main cluster (default: True)
- Use when: You want to discover tables in a business context, then use explore_table() for specific schemas

## Relationship Discovery Tools

**find_path(tables, max_hops=3)**
- Find all connection paths between the given tables (MOST IMPORTANT for JOIN construction)
- tables: Comma-separated list of tables to find connections between
- max_hops: Maximum relationship hops to explore (default: 3)
- Returns: Clean connection map with exact JOIN paths (automatically filters redundant subpaths)
- Use when: You need to construct JOINs between specific tables

**suggest_joins(base_tables, max_suggestions=5, max_hops=1, per_table=False)**
- Suggest additional tables that could be joined with the given base tables
- base_tables: Comma-separated list of base tables to suggest joins for
- max_suggestions: Maximum number of suggestions to return (default: 5)
- max_hops: Maximum relationship hops to explore - 1=direct, 2=two-hop, etc. (default: 1)
- per_table: If True, return suggestions organized per base table; if False, return combined results (default: False)
- Returns: Suggested tables ranked by importance with connection paths
- Use when: You want to explore what other tables might be relevant

**find_related_views(table_names)**
- Find database views related to specific tables for statistics and reporting queries
- table_names: Comma-separated list of table names to find related views for
- Returns: Related statistical views ranked by importance
- Use when: You need statistics/reports based on operational tables

## Recommended Workflow

1. **Start broad**: Use `get_main_cluster()` or `list_clusters()` to understand the database
2. **Focus domain**: Use `show_cluster(cluster_id, detailed=False)` to get table names from specific business areas  
3. **Plan JOINs**: Use `find_path("table1,table2,table3")` to get exact connection paths between target tables
4. **Expand query**: Use `suggest_joins("base_tables")` to discover additional relevant tables
5. **Get details**: Use `explore_table("specific_tables")` for operational schema information
6. **Find statistics**: Use `find_related_views("base_tables")` and `explore_view("view_names")` for reporting/analytics

## Key Principles

- **find_path()** is your primary tool for JOIN construction - shows exact paths with clean output
- **show_cluster() with detailed=False** for discovery, then **explore_table()** for specific schemas
- **find_related_views()** and **explore_view()** for statistics, reporting, and analytics queries
- **suggest_joins()** helps discover relevant tables you might not have considered  
- All tools work with existing Neo4j knowledge graph (no schema rebuilding needed)
- Connection paths show exact table sequences for JOINs (e.g., "user_ - instrument - sample")
- Tools use comma-separated strings for simple, consistent parameter format
- Output is optimized for LLMs - minimal noise, maximum useful information
- Clear separation: tables for operational queries, views for statistics/reporting
```

### Integration Example

```bash
# Start MCP server
python -m src.relational_kg.mcp_server

# Use in LLM applications via MCP protocol
# Tools provide structured JSON responses perfect for LLM processing
```

## Recent Enhancements

✅ **Neo4j Integration**: Full backend support for large-scale schemas  
✅ **LLM Integration**: Automatic cluster analysis and keyword extraction  
✅ **Advanced Clustering**: Importance-based clustering with overlapping support  
✅ **MCP Tools**: Complete toolkit for autonomous LLM database exploration  
✅ **Path Discovery**: Multi-hop relationship mapping for complex JOIN construction  
✅ **Enhanced CLI**: Comprehensive command set with organized categories  
✅ **Optimized MCP Output**: Clean, noise-free responses perfect for LLM consumption  
✅ **Consistent Parameter Format**: Comma-separated strings across all tools  
✅ **Efficient Two-Step Discovery**: Table names first, then selective schema details  
✅ **View Support**: Complete toolkit for statistics and reporting queries with database views  

