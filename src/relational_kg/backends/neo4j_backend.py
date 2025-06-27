from typing import Dict, List, Set, Any, Optional
import logging
from collections import defaultdict

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from .base import GraphBackend
from ..database import TableInfo


class Neo4jBackend(GraphBackend):
    """Neo4j-based graph backend implementation."""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j backend."""
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j package is required for Neo4j backend")
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logger = logging.getLogger(__name__)
        
        # Test connection
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def build_from_schema(self, tables: Dict[str, TableInfo]) -> None:
        """Build the graph from database schema tables."""
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create table and view nodes
            for table_name, table_info in tables.items():
                node_label = "View" if table_info.is_view else "Table"
                session.run(f"""
                    CREATE (t:{node_label} {{
                        name: $name,
                        columns: $columns,
                        primary_keys: $primary_keys,
                        column_count: $column_count,
                        is_view: $is_view,
                        keywords: $keywords,
                        business_concepts: $business_concepts
                    }})
                """, {
                    'name': table_name,
                    'columns': [col.name for col in table_info.columns],
                    'primary_keys': [col.name for col in table_info.columns if col.primary_key],
                    'column_count': len(table_info.columns),
                    'is_view': table_info.is_view,
                    'keywords': [],  # Will be populated by LLM extraction
                    'business_concepts': []  # Will be populated by LLM extraction
                })
            
            # Create relationships
            for table_name, table_info in tables.items():
                # Foreign key relationships for tables
                if not table_info.is_view:
                    for fk in table_info.foreign_keys:
                        target_table = fk['referred_table']
                        if target_table in tables:
                            session.run("""
                                MATCH (source {name: $source})
                                MATCH (target {name: $target})
                                CREATE (source)-[:REFERENCES {
                                    foreign_key_columns: $fk_columns,
                                    strength: $strength,
                                    relationship_type: 'foreign_key'
                                }]->(target)
                            """, {
                                'source': table_name,
                                'target': target_table,
                                'fk_columns': fk['constrained_columns'],
                                'strength': self._calculate_relationship_strength(fk)
                            })
                
                # View dependency relationships
                if table_info.is_view and table_info.view_dependencies:
                    for dependency in table_info.view_dependencies:
                        if dependency in tables:
                            session.run("""
                                MATCH (source {name: $source})
                                MATCH (target {name: $target})
                                CREATE (source)-[:DEPENDS_ON {
                                    strength: 1.0,
                                    relationship_type: 'view_dependency'
                                }]->(target)
                            """, {
                                'source': table_name,
                                'target': dependency
                            })
        
        self.logger.info(f"Built Neo4j graph with {len(tables)} nodes")
    
    def _calculate_relationship_strength(self, foreign_key: Dict[str, Any]) -> float:
        """Calculate strength of relationship based on foreign key properties."""
        strength = 1.0
        
        # Multiple columns increase strength
        if len(foreign_key['constrained_columns']) > 1:
            strength += 0.5
        
        # Cascade operations increase strength
        if foreign_key.get('ondelete') == 'CASCADE':
            strength += 0.3
        if foreign_key.get('onupdate') == 'CASCADE':
            strength += 0.2
            
        return strength
    
    def find_related_tables(self, table_name: str, max_distance: int = 2) -> Set[str]:
        """Find tables/views related to the given table within max_distance hops."""
        with self.driver.session() as session:
            # Use a more compatible query that handles both Tables and Views
            result = session.run("""
                MATCH (start {name: $table_name})
                CALL {
                    WITH start
                    MATCH path = (start)-[*1..%d]-(connected)
                    WHERE connected.name IS NOT NULL
                    RETURN DISTINCT connected.name as table_name
                }
                RETURN DISTINCT table_name
            """ % max_distance, table_name=table_name)
            
            return {record['table_name'] for record in result if record['table_name'] != table_name}
    
    def find_table_clusters(self) -> List[Set[str]]:
        """Find clusters/communities of related tables and views."""
        with self.driver.session() as session:
            # Simple clustering based on connected components
            # Note: This is a basic implementation without GDS library
            result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL
                OPTIONAL MATCH path = (t)-[*]-(connected)
                WHERE connected.name IS NOT NULL
                WITH t, collect(DISTINCT connected.name) + [t.name] as component
                RETURN DISTINCT component
            """)
            
            # Process results to create proper clusters
            all_components = []
            for record in result:
                component = set(record['component'])
                if component not in all_components:
                    all_components.append(component)
            
            # Merge overlapping components
            merged = []
            for component in all_components:
                merged_with_existing = False
                for i, existing in enumerate(merged):
                    if component & existing:  # If there's any overlap
                        merged[i] = existing | component
                        merged_with_existing = True
                        break
                if not merged_with_existing:
                    merged.append(component)
            
            return merged
    
    def get_table_importance(self) -> Dict[str, float]:
        """Get importance scores for all tables and views."""
        with self.driver.session() as session:
            # Calculate importance based on degree centrality
            result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL
                OPTIONAL MATCH (t)-[r]-()
                WITH t, count(r) as degree
                RETURN t.name as table_name, 
                       toFloat(degree) / (SELECT count(*) FROM (MATCH (all) WHERE all.name IS NOT NULL RETURN all)) as importance
            """)
            
            return {record['table_name']: record['importance'] for record in result}
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two tables/views."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start {name: $source})
                MATCH (end {name: $target})
                MATCH path = shortestPath((start)-[*]-(end))
                RETURN [node in nodes(path) | node.name] as path
            """, source=source, target=target)
            
            record = result.single()
            return record['path'] if record else None
    
    def get_table_neighbors(self, table_name: str) -> Set[str]:
        """Get direct neighbors of a table/view."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t {name: $table_name})
                OPTIONAL MATCH (t)-[]-(neighbor)
                WHERE neighbor.name IS NOT NULL
                RETURN DISTINCT neighbor.name as neighbor_name
            """, table_name=table_name)
            
            return {record['neighbor_name'] for record in result if record['neighbor_name']}
    
    def get_all_tables(self) -> Set[str]:
        """Get all table and view names in the graph."""
        with self.driver.session() as session:
            result = session.run("MATCH (t) WHERE t.name IS NOT NULL RETURN t.name as table_name")
            return {record['table_name'] for record in result}
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics (nodes, edges, etc.)."""
        with self.driver.session() as session:
            # Get node count
            node_result = session.run("MATCH (t) WHERE t.name IS NOT NULL RETURN count(t) as node_count")
            node_count = node_result.single()['node_count']
            
            # Get edge count  
            edge_result = session.run("MATCH ()-[r]->() RETURN count(r) as edge_count")
            edge_count = edge_result.single()['edge_count']
            
            # Check connectivity (simplified)
            connectivity_result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL AND NOT EXISTS((t)-[]-())
                RETURN count(t) as isolated_count
            """)
            isolated_count = connectivity_result.single()['isolated_count']
            
            return {
                'node_count': node_count,
                'edge_count': edge_count,
                'density': (edge_count / (node_count * (node_count - 1))) if node_count > 1 else 0,
                'is_connected': isolated_count == 0,
                'isolated_tables': isolated_count
            }
    
    def close(self):
        """Close the Neo4j driver connection."""
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def add_keywords_to_table(self, table_name: str, keywords: List[str], 
                              business_concepts: List[str]) -> None:
        """Add keywords and business concepts to a table/view node."""
        with self.driver.session() as session:
            session.run("""
                MATCH (t {name: $table_name})
                SET t.keywords = $keywords,
                    t.business_concepts = $business_concepts
            """, {
                'table_name': table_name,
                'keywords': keywords,
                'business_concepts': business_concepts
            })
    
    def find_tables_by_keywords(self, search_keywords: List[str], 
                                max_results: int = 10) -> List[Dict[str, Any]]:
        """Find tables/views by matching keywords and business concepts."""
        with self.driver.session() as session:
            # Convert search keywords to lowercase for matching
            search_keywords_lower = [kw.lower() for kw in search_keywords]
            
            result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL
                WITH t,
                     [kw IN t.keywords WHERE ANY(search_kw IN $search_keywords 
                                                WHERE toLower(kw) CONTAINS toLower(search_kw))] as keyword_matches,
                     [bc IN t.business_concepts WHERE ANY(search_kw IN $search_keywords 
                                                         WHERE toLower(bc) CONTAINS toLower(search_kw))] as concept_matches
                WHERE size(keyword_matches) > 0 OR size(concept_matches) > 0
                RETURN t.name as table_name,
                       t.is_view as is_view,
                       keyword_matches,
                       concept_matches,
                       (size(keyword_matches) + size(concept_matches) * 2) as relevance_score
                ORDER BY relevance_score DESC
                LIMIT $max_results
            """, {
                'search_keywords': search_keywords_lower,
                'max_results': max_results
            })
            
            return [
                {
                    'table_name': record['table_name'],
                    'is_view': record['is_view'],
                    'keyword_matches': record['keyword_matches'],
                    'concept_matches': record['concept_matches'],
                    'relevance_score': record['relevance_score']
                }
                for record in result
            ]
    
    def get_all_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """Get all keywords and business concepts for debugging/analysis."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL
                RETURN t.name as table_name,
                       t.keywords as keywords,
                       t.business_concepts as business_concepts
            """)
            
            return {
                record['table_name']: {
                    'keywords': record['keywords'] or [],
                    'business_concepts': record['business_concepts'] or []
                }
                for record in result
            }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()