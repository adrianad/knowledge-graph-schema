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
        """Find clusters/communities of related tables (tables only, excluding views)."""
        with self.driver.session() as session:
            # Get all edges between tables only (exclude views)
            edges_result = session.run("""
                MATCH (a)-[r]-(b)
                WHERE a.name IS NOT NULL AND b.name IS NOT NULL
                AND a.is_view = false AND b.is_view = false
                RETURN DISTINCT a.name as source, b.name as target, 
                       COALESCE(r.strength, 1.0) as weight
            """)
            
            # Get all isolated tables (no edges, and not views)
            isolated_result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL AND t.is_view = false
                AND NOT EXISTS((t)-[]-())
                RETURN t.name as name
            """)
            
            # Build NetworkX graph from Neo4j data
            import networkx as nx
            from networkx.algorithms import community
            
            graph = nx.Graph()
            
            # Add edges between tables
            for record in edges_result:
                graph.add_edge(
                    record['source'], 
                    record['target'], 
                    weight=record['weight']
                )
            
            # Add isolated tables
            for record in isolated_result:
                graph.add_node(record['name'])
            
            if len(graph.nodes) == 0:
                return []
            
            # Use Louvain method for community detection on tables only
            communities = community.louvain_communities(graph)
            return [set(c) for c in communities]
    
    def find_importance_based_clusters(self, min_cluster_size: int = 4, max_hops: int = 2, top_tables_pct: float = 0.2) -> List[Set[str]]:
        """Find clusters based on most important tables as cores."""
        
        # Get table importance scores
        importance = self.get_table_importance()
        
        # Filter to tables only (exclude views)
        with self.driver.session() as session:
            tables_result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL AND t.is_view = false
                RETURN t.name as name
            """)
            table_names = {record['name'] for record in tables_result}
        
        table_importance = {
            table: score for table, score in importance.items()
            if table in table_names
        }
        
        if not table_importance:
            return []
        
        # Find top important tables as cluster cores
        sorted_tables = sorted(table_importance.items(), key=lambda x: x[1], reverse=True)
        num_cores = max(1, int(len(sorted_tables) * top_tables_pct))
        potential_cores = [table for table, _ in sorted_tables[:num_cores]]
        
        clusters = []
        
        for core_table in potential_cores:
            # Build cluster around this core table using Neo4j queries
            cluster = {core_table}
            
            # Find all tables within max_hops of the core
            with self.driver.session() as session:
                cluster_result = session.run("""
                    MATCH (start {name: $core_table})
                    CALL {
                        WITH start
                        MATCH path = (start)-[*1..%d]-(connected)
                        WHERE connected.name IS NOT NULL 
                        AND connected.is_view = false
                        AND connected.name <> $core_table
                        RETURN DISTINCT connected.name as table_name
                    }
                    RETURN DISTINCT table_name
                """ % max_hops, core_table=core_table)
                
                related_tables = {record['table_name'] for record in cluster_result}
                cluster.update(related_tables)
            
            # Only keep clusters that meet minimum size
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        
        return clusters
    
    def get_table_importance(self) -> Dict[str, float]:
        """Get importance scores for all tables and views."""
        with self.driver.session() as session:
            # First get total count of nodes
            total_count_result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL
                RETURN count(t) as total_count
            """)
            total_count = total_count_result.single()['total_count']
            
            if total_count == 0:
                return {}
            
            # Calculate importance based on degree centrality
            result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL
                OPTIONAL MATCH (t)-[r]-()
                WITH t, count(r) as degree
                RETURN t.name as table_name, 
                       toFloat(degree) / $total_count as importance
            """, total_count=total_count)
            
            return {record['table_name']: record['importance'] for record in result}
    
    def get_table_and_view_importance(self) -> Dict[str, Dict[str, Any]]:
        """Get importance scores for tables and views separately."""
        with self.driver.session() as session:
            # First get total count of nodes
            total_count_result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL
                RETURN count(t) as total_count
            """)
            total_count = total_count_result.single()['total_count']
            
            if total_count == 0:
                return {'tables': {}, 'views': {}}
            
            # Get importance for tables
            tables_result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL AND t.is_view = false
                OPTIONAL MATCH (t)-[r]-()
                WITH t, count(r) as degree
                RETURN t.name as table_name, 
                       toFloat(degree) / $total_count as importance
                ORDER BY importance DESC
            """, total_count=total_count)
            
            # Get importance for views (simplified to avoid hanging)
            views_result = session.run("""
                MATCH (v)
                WHERE v.name IS NOT NULL AND v.is_view = true
                OPTIONAL MATCH (v)-[r]-()
                WITH v, count(r) as degree
                RETURN v.name as table_name, 
                       toFloat(degree) / $total_count as importance
                ORDER BY importance DESC
            """, total_count=total_count)
            
            tables = {record['table_name']: record['importance'] for record in tables_result}
            views = {record['table_name']: record['importance'] for record in views_result}
            
            return {
                'tables': tables,
                'views': views
            }
    
    def find_shortest_path(self, source: str, target: str, max_hops: Optional[int] = None) -> Optional[List[str]]:
        """Find shortest path between two tables/views."""
        with self.driver.session() as session:
            # Use reasonable default if no max_hops specified
            hop_limit = max_hops if max_hops is not None else 10
            
            result = session.run(f"""
                MATCH (start {{name: $source}})
                MATCH (end {{name: $target}})
                MATCH path = shortestPath((start)-[*..{hop_limit}]-(end))
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
    
    def find_tables_and_views_by_keywords(self, search_keywords: List[str], 
                                          max_tables: int = 5, max_views: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Find tables and views separately by matching keywords and business concepts."""
        with self.driver.session() as session:
            # Convert search keywords to lowercase for matching
            search_keywords_lower = [kw.lower() for kw in search_keywords]
            
            # Get top tables (is_view = false)
            tables_result = session.run("""
                MATCH (t)
                WHERE t.name IS NOT NULL AND t.is_view = false
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
                LIMIT $max_tables
            """, {
                'search_keywords': search_keywords_lower,
                'max_tables': max_tables
            })
            
            # Get top views (is_view = true)
            views_result = session.run("""
                MATCH (v)
                WHERE v.name IS NOT NULL AND v.is_view = true
                WITH v,
                     [kw IN v.keywords WHERE ANY(search_kw IN $search_keywords 
                                                WHERE toLower(kw) CONTAINS toLower(search_kw))] as keyword_matches,
                     [bc IN v.business_concepts WHERE ANY(search_kw IN $search_keywords 
                                                         WHERE toLower(bc) CONTAINS toLower(search_kw))] as concept_matches
                WHERE size(keyword_matches) > 0 OR size(concept_matches) > 0
                RETURN v.name as table_name,
                       v.is_view as is_view,
                       keyword_matches,
                       concept_matches,
                       (size(keyword_matches) + size(concept_matches) * 2) as relevance_score
                ORDER BY relevance_score DESC
                LIMIT $max_views
            """, {
                'search_keywords': search_keywords_lower,
                'max_views': max_views
            })
            
            tables = [
                {
                    'table_name': record['table_name'],
                    'is_view': record['is_view'],
                    'keyword_matches': record['keyword_matches'],
                    'concept_matches': record['concept_matches'],
                    'relevance_score': record['relevance_score']
                }
                for record in tables_result
            ]
            
            views = [
                {
                    'table_name': record['table_name'],
                    'is_view': record['is_view'],
                    'keyword_matches': record['keyword_matches'],
                    'concept_matches': record['concept_matches'],
                    'relevance_score': record['relevance_score']
                }
                for record in views_result
            ]
            
            return {
                'tables': tables,
                'views': views
            }
    
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
    
    def store_table_clusters_with_analysis(self, clusters: List[Set[str]], cluster_analyses: List[Any]) -> None:
        """Store table clusters with LLM-generated names and descriptions."""
        with self.driver.session() as session:
            # Clear existing cluster data
            session.run("""
                MATCH (c:Cluster)
                DETACH DELETE c
            """)
            
            # Remove existing BELONGS_TO_CLUSTER relationships (if any leftover)
            session.run("""
                MATCH ()-[r:BELONGS_TO_CLUSTER]-()
                DELETE r
            """)
            
            for cluster_tables, analysis in zip(clusters, cluster_analyses):
                if len(cluster_tables) == 0:
                    continue
                    
                cluster_tables_list = list(cluster_tables)  # Convert set to list for Neo4j
                
                # Calculate cluster metadata
                cluster_size = len(cluster_tables_list)
                connectivity_score = self._calculate_cluster_connectivity(cluster_tables_list)
                
                # Create cluster node with LLM analysis
                session.run("""
                    CREATE (c:Cluster {
                        id: $cluster_id,
                        name: $name,
                        description: $description,
                        business_domain: $business_domain,
                        size: $size,
                        connectivity_score: $connectivity_score,
                        keywords: $keywords,
                        confidence: $confidence,
                        created_at: timestamp()
                    })
                """, {
                    'cluster_id': analysis.cluster_id,
                    'name': analysis.name,
                    'description': analysis.description,
                    'business_domain': analysis.business_domain,
                    'size': cluster_size,
                    'connectivity_score': connectivity_score,
                    'keywords': analysis.keywords,
                    'confidence': analysis.confidence
                })
                
                # Create relationships from tables to cluster
                for table_name in cluster_tables_list:
                    session.run("""
                        MATCH (t {name: $table_name})
                        MATCH (c:Cluster {id: $cluster_id})
                        CREATE (t)-[:BELONGS_TO_CLUSTER]->(c)
                    """, {
                        'table_name': table_name,
                        'cluster_id': analysis.cluster_id
                    })
                
                self.logger.debug(f"Created cluster '{analysis.name}' ({analysis.cluster_id}) with {cluster_size} tables")
            
            self.logger.info(f"Stored {len(clusters)} analyzed table clusters in Neo4j")
    
    def store_table_clusters(self, clusters: List[Set[str]]) -> None:
        """Store table clusters as Cluster nodes with relationships to tables (legacy method)."""
        with self.driver.session() as session:
            # Clear existing cluster data
            session.run("""
                MATCH (c:Cluster)
                DETACH DELETE c
            """)
            
            # Remove existing BELONGS_TO_CLUSTER relationships (if any leftover)
            session.run("""
                MATCH ()-[r:BELONGS_TO_CLUSTER]-()
                DELETE r
            """)
            
            for i, cluster_tables in enumerate(clusters, 1):
                if len(cluster_tables) == 0:
                    continue
                    
                cluster_id = f"cluster_{i}"
                cluster_tables_list = list(cluster_tables)  # Convert set to list for Neo4j
                
                # Calculate cluster metadata
                cluster_size = len(cluster_tables_list)
                connectivity_score = self._calculate_cluster_connectivity(cluster_tables_list)
                domain_keywords = self._get_cluster_keywords(cluster_tables_list)
                
                # Create cluster node
                session.run("""
                    CREATE (c:Cluster {
                        id: $cluster_id,
                        name: $cluster_id,
                        description: $description,
                        size: $size,
                        connectivity_score: $connectivity_score,
                        keywords: $keywords,
                        created_at: timestamp()
                    })
                """, {
                    'cluster_id': cluster_id,
                    'description': f"Database cluster containing {cluster_size} related tables",
                    'size': cluster_size,
                    'connectivity_score': connectivity_score,
                    'keywords': domain_keywords
                })
                
                # Create relationships from tables to cluster
                for table_name in cluster_tables_list:
                    session.run("""
                        MATCH (t {name: $table_name})
                        MATCH (c:Cluster {id: $cluster_id})
                        CREATE (t)-[:BELONGS_TO_CLUSTER]->(c)
                    """, {
                        'table_name': table_name,
                        'cluster_id': cluster_id
                    })
                
                self.logger.debug(f"Created cluster {cluster_id} with {cluster_size} tables")
            
            self.logger.info(f"Stored {len(clusters)} table clusters in Neo4j")
    
    def _calculate_cluster_connectivity(self, cluster_tables: List[str]) -> float:
        """Calculate internal connectivity score for a cluster."""
        if len(cluster_tables) <= 1:
            return 1.0
            
        with self.driver.session() as session:
            # Count internal edges within the cluster
            result = session.run("""
                MATCH (a)-[r]-(b)
                WHERE a.name IN $cluster_tables 
                AND b.name IN $cluster_tables
                AND a.name <> b.name
                RETURN count(DISTINCT r) as internal_edges
            """, cluster_tables=cluster_tables)
            
            internal_edges = result.single()['internal_edges']
            
            # Maximum possible edges in a cluster (undirected graph)
            max_possible_edges = len(cluster_tables) * (len(cluster_tables) - 1) / 2
            
            # Return connectivity score (0 to 1)
            return internal_edges / max_possible_edges if max_possible_edges > 0 else 0.0
    
    def _get_cluster_keywords(self, cluster_tables: List[str]) -> List[str]:
        """Get aggregated keywords from all tables in a cluster."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t)
                WHERE t.name IN $cluster_tables
                RETURN t.keywords as keywords, t.business_concepts as business_concepts
            """, cluster_tables=cluster_tables)
            
            all_keywords = set()
            for record in result:
                keywords = record['keywords'] or []
                business_concepts = record['business_concepts'] or []
                all_keywords.update(keywords)
                all_keywords.update(business_concepts)
            
            return sorted(list(all_keywords))

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()