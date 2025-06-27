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

2. **Find relevant tables for your query:**
```bash
rkg find-tables -c "sqlite:///your_database.db" -k "user,order,product"
```

3. **Create intelligent table clusters:**
```bash
rkg create-clusters -c "sqlite:///your_database.db" --method importance
```

4. **Visualize the schema graph:**
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

# Semantic table search with natural language
rkg find-tables-semantic -c "sqlite:///shop.db" \
  -q "tables related to customer orders and payments"

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

## Recent Enhancements

✅ **Neo4j Integration**: Full backend support for large-scale schemas  
✅ **LLM Integration**: Automatic cluster analysis and keyword extraction  
✅ **Advanced Clustering**: Importance-based clustering with overlapping support  
✅ **Semantic Search**: Natural language table discovery  
✅ **Enhanced CLI**: Comprehensive command set with flexible options  

## Future Roadmap

- Advanced relationship strength scoring based on data analysis
- Schema evolution tracking and change detection  
- Web API interface for programmatic access
- Machine learning models for query-to-table mapping
- Support for additional database types (Oracle, SQL Server)
- Real-time schema monitoring and alerts