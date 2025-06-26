"""Knowledge graph construction and analysis."""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any, Optional
import json
import logging

import networkx as nx
from networkx.algorithms import community

from .database import TableInfo, ColumnInfo


@dataclass
class GraphNode:
    """Graph node representing a table."""
    table_name: str
    columns: List[str]
    primary_keys: List[str]
    column_count: int
    

@dataclass
class GraphEdge:
    """Graph edge representing a relationship."""
    source: str
    target: str
    relationship_type: str
    foreign_key_columns: List[str]
    strength: float = 1.0


class SchemaGraph:
    """Knowledge graph for database schema."""
    
    def __init__(self):
        """Initialize empty graph."""
        self.graph = nx.DiGraph()
        self.tables: Dict[str, TableInfo] = {}
        self.logger = logging.getLogger(__name__)
    
    def build_from_schema(self, tables: Dict[str, TableInfo]) -> None:
        """Build graph from extracted schema information."""
        self.tables = tables
        self._add_table_nodes()
        self._add_relationship_edges()
        self.logger.info(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
    def _add_table_nodes(self) -> None:
        """Add table nodes to the graph."""
        for table_name, table_info in self.tables.items():
            columns = [col.name for col in table_info.columns]
            primary_keys = [col.name for col in table_info.columns if col.primary_key]
            
            node_data = GraphNode(
                table_name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                column_count=len(columns)
            )
            
            self.graph.add_node(table_name, **node_data.__dict__)
    
    def _add_relationship_edges(self) -> None:
        """Add relationship edges based on foreign keys."""
        for table_name, table_info in self.tables.items():
            for fk in table_info.foreign_keys:
                target_table = fk['referred_table']
                if target_table in self.tables:
                    
                    edge_data = GraphEdge(
                        source=table_name,
                        target=target_table,
                        relationship_type='foreign_key',
                        foreign_key_columns=fk['constrained_columns'],
                        strength=self._calculate_relationship_strength(fk)
                    )
                    
                    self.graph.add_edge(
                        table_name, 
                        target_table, 
                        **edge_data.__dict__
                    )
    
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
        """Find tables related to given table within max distance."""
        if table_name not in self.graph:
            return set()
        
        related = set()
        
        # Use BFS to find related tables
        for distance in range(1, max_distance + 1):
            # Outgoing relationships (this table references others)
            try:
                successors = nx.single_source_shortest_path_length(
                    self.graph, table_name, cutoff=distance
                )
                related.update(successors.keys())
            except nx.NetworkXError:
                pass
            
            # Incoming relationships (others reference this table)
            try:
                predecessors = nx.single_source_shortest_path_length(
                    self.graph.reverse(), table_name, cutoff=distance
                )
                related.update(predecessors.keys())
            except nx.NetworkXError:
                pass
        
        related.discard(table_name)  # Remove the original table
        return related
    
    def find_table_clusters(self) -> List[Set[str]]:
        """Find communities/clusters of related tables."""
        # Convert to undirected graph for community detection
        undirected = self.graph.to_undirected()
        
        if len(undirected.nodes) == 0:
            return []
        
        # Use Louvain method for community detection
        communities = community.louvain_communities(undirected)
        return [set(c) for c in communities]
    
    def get_table_importance(self) -> Dict[str, float]:
        """Calculate importance scores for tables based on centrality."""
        if len(self.graph.nodes) == 0:
            return {}
        
        # PageRank for importance
        pagerank = nx.pagerank(self.graph)
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Combine scores
        importance = {}
        for table in self.graph.nodes:
            importance[table] = (
                pagerank.get(table, 0) * 0.7 + 
                degree_centrality.get(table, 0) * 0.3
            )
        
        return importance
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two tables."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_table_neighbors(self, table_name: str) -> Dict[str, List[str]]:
        """Get direct neighbors of a table."""
        if table_name not in self.graph:
            return {'predecessors': [], 'successors': []}
        
        return {
            'predecessors': list(self.graph.predecessors(table_name)),
            'successors': list(self.graph.successors(table_name))
        }
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph structure as JSON-serializable data."""
        nodes = []
        edges = []
        
        for node, data in self.graph.nodes(data=True):
            nodes.append({
                'id': node,
                **data
            })
        
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                **data
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'node_count': len(self.graph.nodes),
                'edge_count': len(self.graph.edges),
                'is_connected': nx.is_weakly_connected(self.graph)
            }
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save graph data to JSON file."""
        data = self.export_graph_data()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Graph saved to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if len(self.graph.nodes) == 0:
            return {'node_count': 0, 'edge_count': 0}
        
        return {
            'node_count': len(self.graph.nodes),
            'edge_count': len(self.graph.edges),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'strongly_connected_components': len(list(nx.strongly_connected_components(self.graph))),
            'average_clustering': nx.average_clustering(self.graph.to_undirected())
        }