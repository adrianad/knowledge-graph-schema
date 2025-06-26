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
- Knowledge graph construction with table importance scoring
- Table relevance search based on keywords
- Community detection for related table clusters
- Join suggestions based on graph relationships
- Interactive HTML visualizations
- Schema subset export for LLM integration

## CLI Commands
- `rkg analyze -c "connection_string"` - Full schema analysis
- `rkg find-tables -c "connection_string" -k "keywords"` - Keyword-based table search  
- `rkg suggest-joins -c "connection_string" -t "tables"` - Join recommendations
- `rkg visualize -c "connection_string"` - Interactive graph visualization
- `rkg summary -c "connection_string"` - Schema statistics

## Development Status
✅ Phase 1 Complete: Core infrastructure, schema extraction, graph building, CLI
🔄 Future: Neo4j integration (architecture ready - see NEO4J_INTEGRATION.md)

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