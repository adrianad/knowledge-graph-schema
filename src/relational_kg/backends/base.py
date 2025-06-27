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
    def get_table_importance(self) -> Dict[str, float]:
        """Get importance scores for all tables."""
        pass
    
    @abstractmethod
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
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