# Claude Code Memory

## Project Overview
This is a Python application that creates knowledge graphs from database schemas to help LLMs understand table relationships for more efficient SQL generation.

## Architecture
- **DatabaseExtractor**: SQLAlchemy 2.x based schema extraction (SQLite, PostgreSQL, MySQL)
- **SchemaGraph**: NetworkX knowledge graph with relationship analysis
- **SchemaAnalyzer**: Intelligent table discovery and relevance scoring
- **GraphVisualizer**: Interactive Plotly and static matplotlib visualizations
- **CLI**: Full-featured command-line interface

## Key Features Implemented
- Database schema extraction with foreign key relationships
- Knowledge graph construction with table importance scoring (PageRank + degree centrality)
- Table relevance search based on keywords and semantic analysis
- Advanced clustering methods:
  - Community detection for related table clusters (Louvain algorithm)
  - Importance-based clustering using high-importance tables as cores
  - Overlapping clusters (tables can belong to multiple clusters)
- LLM-powered cluster analysis with intelligent naming and keyword generation
- Join suggestions based on graph relationships
- Interactive HTML visualizations
- Schema subset export for LLM integration
- Neo4j integration for large-scale schemas and persistence

## CLI Commands
- `rkg analyze -c "connection_string"` - Full schema analysis
- `rkg find-tables -c "connection_string" -k "keywords"` - Keyword-based table search  
- `rkg suggest-joins -c "connection_string" -t "tables"` - Join recommendations
- `rkg visualize -c "connection_string"` - Interactive graph visualization
- `rkg summary -c "connection_string"` - Schema statistics
- `rkg create-clusters -c "connection_string"` - Generate and store table clusters with LLM analysis
  - `--method community` - Community detection clustering (default)
  - `--method importance` - Importance-based clustering around hub tables
  - `--min-size N` - Minimum cluster size (default: 4)
  - `--max-hops N` - Maximum relationship distance (default: 2)
  - `--top-pct 0.2` - Percentage of top tables to use as cores (default: 20%)
- `rkg llm-keyword-extraction -c "connection_string"` - Extract business keywords using LLM
- `rkg find-tables-semantic -c "connection_string" -q "natural language query"` - Semantic table search
- `rkg explore-table -c "connection_string" -t "table_name"` - Explore relationships from a specific table

## Development Status
✅ Phase 1 Complete: Core infrastructure, schema extraction, graph building, CLI
✅ Phase 2 Complete: Advanced clustering, LLM integration, Neo4j backend
🔄 Current: Enhanced clustering algorithms and semantic analysis

## Clustering Methods
### Community Detection (Traditional)
- Uses Louvain algorithm for community detection
- Groups tables based on natural relationship patterns
- Good for discovering organic clusters in the schema

### Importance-Based Clustering (New)
- Uses table importance scores (PageRank + degree centrality) to identify hub tables
- Builds clusters around the most important tables as cores
- Supports overlapping clusters - tables can belong to multiple clusters
- Configurable parameters for cluster size, relationship distance, and core selection
- Better for understanding business domains centered around key entities

## LLM Integration
- Automatic cluster naming and description generation
- Business keyword extraction for tables and clusters
- Supports local LLM servers (qwen, llama, etc.) via OpenAI-compatible API
- Enhanced prompts with table column information for better context
- Fallback mechanisms for when LLM calls fail

## Use Case
The primary goal is schema reduction for LLMs - instead of providing entire database schemas to LLMs for SQL generation, this tool identifies relevant tables based on query intent and provides focused schema subsets.

## Neo4j Integration Notes
- Neo4j requires a running server (local, cloud, or Docker)
- Advantages: Better performance for large schemas (>1000 tables), persistence, advanced graph queries
- Current NetworkX approach is sufficient for most use cases (<1000 tables)
- Architecture supports easy backend switching when needed
- Full integration plan documented in NEO4J_INTEGRATION.md

## Installation & Usage
```bash
pip install -e .
rkg analyze -c "sqlite:///your_database.db"
rkg find-tables -c "connection_string" -k "user,order,product"
```

## Dependencies
Uses latest versions: SQLAlchemy 2.x, NetworkX 3.2+, Plotly 5.17+, Click 8.1+