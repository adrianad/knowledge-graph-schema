from abc import ABC, abstractmethod
from typing import Dict, List, Set, Any, Optional
from ..database import TableInfo


class GraphBackend(ABC):
    """Abstract base class for graph backend implementations."""
    
    @abstractmethod
    def build_from_schema(self, tables: Dict[str, TableInfo]) -> None:
        """Build the graph from database schema tables."""
        pass
    
    @abstractmethod
    def find_related_tables(self, table_name: str, max_distance: int = 2) -> Set[str]:
        """Find tables related to the given table within max_distance hops."""
        pass
    
    @abstractmethod
    def find_table_clusters(self) -> List[Set[str]]:
        """Find clusters/communities of related tables."""
        pass
    
    @abstractmethod
    def find_importance_based_clusters(self, min_cluster_size: int = 4, max_hops: int = 2, top_tables_pct: float = 0.2) -> List[Set[str]]:
        """Find clusters based on most important tables as cores."""
        pass
    
    @abstractmethod
    def get_table_importance(self) -> Dict[str, float]:
        """Get importance scores for all tables."""
        pass
    
    @abstractmethod
    def find_shortest_path(self, source: str, target: str, max_hops: Optional[int] = None) -> Optional[List[str]]:
        """Find shortest path between two tables."""
        pass
    
    @abstractmethod
    def get_table_neighbors(self, table_name: str) -> Set[str]:
        """Get direct neighbors of a table."""
        pass
    
    @abstractmethod
    def get_all_tables(self) -> Set[str]:
        """Get all table names in the graph."""
        pass
    
    @abstractmethod
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics (nodes, edges, etc.)."""
        pass
    
    @abstractmethod
    def get_all_clusters(self) -> List[Dict[str, Any]]:
        """Get basic information about all clusters."""
        pass
    
    @abstractmethod
    def get_cluster_tables(self, cluster_id: str, detailed: bool = False, connection_string: str = None) -> List[Dict[str, Any]]:
        """Get detailed table information for a specific cluster."""
        pass
    
    @abstractmethod
    def get_table_details(self, table_names: List[str], detailed: bool = True, connection_string: str = None) -> List[Dict[str, Any]]:
        """Get detailed information for specific tables."""
        pass
    
    @abstractmethod
    def get_tables_for_keyword_extraction(self, connection_string: str, include_views: bool = True) -> Dict[str, Any]:
        """Get tables that need keyword extraction and their detailed schema information."""
        pass