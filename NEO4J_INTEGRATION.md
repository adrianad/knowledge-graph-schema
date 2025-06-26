# Neo4j Integration Plan

## Overview
This document outlines how to integrate Neo4j as an alternative graph backend to the existing NetworkX implementation.

## Prerequisites

### Neo4j Setup Options
1. **Local Development**: Neo4j Desktop or Community Server
2. **Docker**: `docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j`
3. **Cloud**: Neo4j Aura (managed service)
4. **Self-hosted**: Install Neo4j on your server

### Required Dependencies
Add to `requirements.txt`:
```
neo4j>=5.0.0
```

## Architecture Changes

### 1. Abstract Graph Backend Interface

Create `src/relational_kg/backends/base.py`:
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Any, Optional
from ..database import TableInfo

class GraphBackend(ABC):
    @abstractmethod
    def build_from_schema(self, tables: Dict[str, TableInfo]) -> None:
        pass
    
    @abstractmethod
    def find_related_tables(self, table_name: str, max_distance: int = 2) -> Set[str]:
        pass
    
    @abstractmethod
    def find_table_clusters(self) -> List[Set[str]]:
        pass
    
    @abstractmethod
    def get_table_importance(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        pass
```

### 2. NetworkX Backend Implementation

Move current logic to `src/relational_kg/backends/networkx_backend.py`:
```python
from .base import GraphBackend
from ..graph import SchemaGraph

class NetworkXBackend(GraphBackend):
    def __init__(self):
        self.schema_graph = SchemaGraph()
    
    def build_from_schema(self, tables: Dict[str, TableInfo]) -> None:
        self.schema_graph.build_from_schema(tables)
    
    # ... implement all abstract methods
```

### 3. Neo4j Backend Implementation

Create `src/relational_kg/backends/neo4j_backend.py`:
```python
from neo4j import GraphDatabase
from .base import GraphBackend

class Neo4jBackend(GraphBackend):
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def build_from_schema(self, tables: Dict[str, TableInfo]) -> None:
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create table nodes
            for table_name, table_info in tables.items():
                session.run("""
                    CREATE (t:Table {
                        name: $name,
                        columns: $columns,
                        primary_keys: $primary_keys,
                        column_count: $column_count
                    })
                """, {
                    'name': table_name,
                    'columns': [col.name for col in table_info.columns],
                    'primary_keys': [col.name for col in table_info.columns if col.primary_key],
                    'column_count': len(table_info.columns)
                })
            
            # Create relationships
            for table_name, table_info in tables.items():
                for fk in table_info.foreign_keys:
                    target_table = fk['referred_table']
                    if target_table in tables:
                        session.run("""
                            MATCH (source:Table {name: $source})
                            MATCH (target:Table {name: $target})
                            CREATE (source)-[:REFERENCES {
                                foreign_key_columns: $fk_columns,
                                strength: $strength
                            }]->(target)
                        """, {
                            'source': table_name,
                            'target': target_table,
                            'fk_columns': fk['constrained_columns'],
                            'strength': self._calculate_relationship_strength(fk)
                        })
    
    def find_related_tables(self, table_name: str, max_distance: int = 2) -> Set[str]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:Table {name: $table_name})
                CALL apoc.path.subgraphAll(start, {
                    maxLevel: $max_distance,
                    relationshipFilter: "REFERENCES|<REFERENCES"
                })
                YIELD nodes
                UNWIND nodes as node
                RETURN DISTINCT node.name as table_name
            """, table_name=table_name, max_distance=max_distance)
            
            return {record['table_name'] for record in result if record['table_name'] != table_name}
    
    def find_table_clusters(self) -> List[Set[str]]:
        with self.driver.session() as session:
            result = session.run("""
                CALL gds.louvain.stream('myGraph')
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId).name as table_name, communityId
            """)
            
            clusters = {}
            for record in result:
                community_id = record['communityId']
                table_name = record['table_name']
                if community_id not in clusters:
                    clusters[community_id] = set()
                clusters[community_id].add(table_name)
            
            return list(clusters.values())
    
    def get_table_importance(self) -> Dict[str, float]:
        with self.driver.session() as session:
            result = session.run("""
                CALL gds.pageRank.stream('myGraph')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name as table_name, score
            """)
            
            return {record['table_name']: record['score'] for record in result}
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:Table {name: $source})
                MATCH (end:Table {name: $target})
                MATCH path = shortestPath((start)-[*]-(end))
                RETURN [node in nodes(path) | node.name] as path
            """, source=source, target=target)
            
            record = result.single()
            return record['path'] if record else None
    
    def close(self):
        self.driver.close()
```

### 4. Updated SchemaAnalyzer

Modify `src/relational_kg/analyzer.py`:
```python
class SchemaAnalyzer:
    def __init__(self, connection_string: str, backend: str = 'networkx', **backend_kwargs):
        self.extractor = DatabaseExtractor(connection_string)
        
        if backend == 'networkx':
            from .backends.networkx_backend import NetworkXBackend
            self.graph_backend = NetworkXBackend()
        elif backend == 'neo4j':
            from .backends.neo4j_backend import Neo4jBackend
            self.graph_backend = Neo4jBackend(**backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def analyze_schema(self) -> None:
        self.extractor.connect()
        tables = self.extractor.extract_schema()
        self.graph_backend.build_from_schema(tables)
    
    # Update all methods to use self.graph_backend instead of self.graph
```

### 5. CLI Integration

Update `src/relational_kg/cli.py`:
```python
@click.option('--backend', '-b', default='networkx', help='Graph backend (networkx, neo4j)')
@click.option('--neo4j-uri', help='Neo4j connection URI')
@click.option('--neo4j-user', help='Neo4j username')
@click.option('--neo4j-password', help='Neo4j password')
def analyze(connection: str, backend: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, output: Optional[str]) -> None:
    """Analyze database schema with specified backend."""
    backend_kwargs = {}
    if backend == 'neo4j':
        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            click.echo("Neo4j backend requires --neo4j-uri, --neo4j-user, and --neo4j-password")
            sys.exit(1)
        backend_kwargs = {
            'uri': neo4j_uri,
            'username': neo4j_user,
            'password': neo4j_password
        }
    
    analyzer = SchemaAnalyzer(connection, backend=backend, **backend_kwargs)
    # ... rest of the function
```

## Usage Examples

### With NetworkX (current):
```bash
rkg analyze -c "sqlite:///shop.db"
```

### With Neo4j:
```bash
rkg analyze -c "sqlite:///shop.db" \
    --backend neo4j \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password
```

## Cypher Query Examples

### Find tables related to 'users' within 3 hops:
```cypher
MATCH (start:Table {name: 'users'})
CALL apoc.path.subgraphAll(start, {maxLevel: 3})
YIELD nodes
UNWIND nodes as node
RETURN DISTINCT node.name
```

### Find the most connected tables:
```cypher
MATCH (t:Table)
RETURN t.name, size((t)-[:REFERENCES]-()) as connections
ORDER BY connections DESC
LIMIT 10
```

### Community detection:
```cypher
CALL gds.louvain.stream('myGraph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name as table_name, communityId
ORDER BY communityId
```

## Performance Comparison

| Feature | NetworkX | Neo4j |
|---------|----------|--------|
| Setup | Simple | Requires server |
| Memory Usage | High (in-memory) | Low (disk-based) |
| Query Speed | Fast (<1k nodes) | Fast (any size) |
| Scalability | Limited | Excellent |
| Persistence | Manual | Automatic |
| Query Language | Python API | Cypher |

## Migration Strategy

1. **Phase 1**: Implement abstract backend interface
2. **Phase 2**: Move NetworkX logic to backend
3. **Phase 3**: Implement Neo4j backend
4. **Phase 4**: Add CLI options for backend selection
5. **Phase 5**: Update documentation and examples

## Benefits of Neo4j Integration

1. **Scalability**: Handle enterprise databases with 10k+ tables
2. **Persistence**: Graph survives application restarts
3. **Advanced Queries**: Complex relationship traversals
4. **Performance**: Optimized graph algorithms
5. **Concurrent Access**: Multiple users/applications
6. **Monitoring**: Built-in graph analytics and monitoring