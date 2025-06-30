from typing import Dict, List, Set, Any, Optional
import logging

from .base import GraphBackend
from ..graph import SchemaGraph
from ..database import TableInfo


class NetworkXBackend(GraphBackend):
    """NetworkX-based graph backend implementation."""
    
    def __init__(self):
        """Initialize NetworkX backend."""
        self.schema_graph = SchemaGraph()
        self.logger = logging.getLogger(__name__)
    
    def build_from_schema(self, tables: Dict[str, TableInfo]) -> None:
        """Build the graph from database schema tables."""
        self.schema_graph.build_from_schema(tables)
    
    def find_related_tables(self, table_name: str, max_distance: int = 2) -> Set[str]:
        """Find tables related to the given table within max_distance hops."""
        return self.schema_graph.find_related_tables(table_name, max_distance)
    
    def find_table_clusters(self) -> List[Set[str]]:
        """Find clusters/communities of related tables."""
        return self.schema_graph.find_table_clusters()
    
    def find_importance_based_clusters(self, min_cluster_size: int = 4, max_hops: int = 2, top_tables_pct: float = 0.2) -> List[Set[str]]:
        """Find clusters based on most important tables as cores."""
        return self.schema_graph.find_importance_based_clusters(min_cluster_size, max_hops, top_tables_pct)
    
    def get_table_importance(self) -> Dict[str, float]:
        """Get importance scores for all tables."""
        return self.schema_graph.get_table_importance()
    
    def get_table_and_view_importance(self) -> Dict[str, Dict[str, Any]]:
        """Get importance scores for tables and views separately."""
        return self.schema_graph.get_table_and_view_importance()
    
    def find_shortest_path(self, source: str, target: str, max_hops: Optional[int] = None) -> Optional[List[str]]:
        """Find shortest path between two tables."""
        return self.schema_graph.find_shortest_path(source, target)
    
    def get_table_neighbors(self, table_name: str) -> Set[str]:
        """Get direct neighbors of a table."""
        neighbors_dict = self.schema_graph.get_table_neighbors(table_name)
        return set(neighbors_dict['predecessors'] + neighbors_dict['successors'])
    
    def get_all_tables(self) -> Set[str]:
        """Get all table names in the graph."""
        return set(self.schema_graph.graph.nodes())
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics (nodes, edges, etc.)."""
        return self.schema_graph.get_statistics()
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph structure as JSON-serializable data."""
        return self.schema_graph.export_graph_data()
    
    def save_to_file(self, filepath: str) -> None:
        """Save graph data to JSON file."""
        self.schema_graph.save_to_file(filepath)
    
    def get_all_clusters(self) -> List[Dict[str, Any]]:
        """Get basic information about all clusters."""
        # NetworkX doesn't persistently store clusters, so return empty list
        return []
    
    def get_cluster_tables(self, cluster_id: str, detailed: bool = False, connection_string: str = None) -> List[Dict[str, Any]]:
        """Get detailed table information for a specific cluster."""
        # NetworkX doesn't persistently store clusters, so return empty list
        return []
    
    def get_table_details(self, table_names: List[str], detailed: bool = True, connection_string: str = None) -> List[Dict[str, Any]]:
        """Get detailed information for specific tables."""
        # NetworkX doesn't have persistent table storage, so return empty list
        return []
    
    def get_tables_for_keyword_extraction(self, connection_string: str, include_views: bool = True) -> Dict[str, Any]:
        """Get tables that need keyword extraction and their detailed schema information."""
        # NetworkX doesn't have persistent keyword storage, so return empty dict
        return {}