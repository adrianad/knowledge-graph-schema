"""Schema analysis and table relationship discovery."""

from typing import Dict, List, Set, Tuple, Any, Optional
import logging
from dataclasses import dataclass

from .database import DatabaseExtractor, TableInfo
from .backends.base import GraphBackend




class SchemaAnalyzer:
    """Analyze database schema and provide intelligent table suggestions."""
    
    def __init__(self, connection_string: str, backend: str = 'neo4j', **backend_kwargs):
        """Initialize analyzer with database connection and Neo4j backend."""
        self.extractor = DatabaseExtractor(connection_string)
        self.logger = logging.getLogger(__name__)
        self._connected = False
        self.tables: Dict[str, TableInfo] = {}
        
        # Initialize Neo4j backend
        if backend != 'neo4j':
            raise ValueError(f"Only 'neo4j' backend is supported, got: {backend}")
            
        from .backends.neo4j_backend import Neo4jBackend
        self.backend = Neo4jBackend(**backend_kwargs)
        
        self.logger.info("Initialized SchemaAnalyzer with Neo4j backend")
    
    def analyze_schema(self, include_views: bool = True) -> None:
        """Analyze database schema and build knowledge graph."""
        self.extractor.connect()
        self._connected = True
        
        # Extract schema information
        self.tables = self.extractor.extract_schema(include_views=include_views)
        
        # Build knowledge graph using backend
        self.backend.build_from_schema(self.tables)
        
        self.logger.info("Schema analysis completed")
    
    
    def get_table_cluster(self, table_name: str) -> Set[str]:
        """Get cluster of related tables for a given table."""
        if not self._connected:
            raise RuntimeError("Schema not analyzed. Call analyze_schema() first.")
        
        # Find tables within distance 2
        related = self.backend.find_related_tables(table_name, max_distance=2)
        related.add(table_name)  # Include the original table
        
        return related
    
    def suggest_tables_for_join(
        self, 
        base_tables: List[str], 
        max_suggestions: int = 5
    ) -> List[str]:
        """Suggest additional tables that could be joined with base tables."""
        # Neo4j backend can work directly without schema analysis
        suggestions = set()
        
        for table in base_tables:
            # Get neighbors (direct relationships) from Neo4j
            neighbors = self.backend.get_table_neighbors(table)
            suggestions.update(neighbors)
        
        # Remove base tables from suggestions
        suggestions = suggestions - set(base_tables)
        
        # Score suggestions by importance
        importance_scores = self.backend.get_table_importance()
        
        # Sort by importance and return top suggestions
        sorted_suggestions = sorted(
            suggestions, 
            key=lambda x: importance_scores.get(x, 0), 
            reverse=True
        )
        
        return sorted_suggestions[:max_suggestions]
    
    def find_connection_path(self, table1: str, table2: str, max_hops: Optional[int] = None) -> Optional[List[str]]:
        """Find connection path between two tables."""
        # Neo4j backend can work directly without schema analysis
        return self.backend.find_shortest_path(table1, table2, max_hops)
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get summary of the database schema."""
        if not self._connected:
            raise RuntimeError("Schema not analyzed. Call analyze_schema() first.")
        
        stats = self.backend.get_graph_stats()
        clusters = self.backend.find_table_clusters()
        
        # Check if backend supports separate table/view importance
        if hasattr(self.backend, 'get_table_and_view_importance'):
            importance_data = self.backend.get_table_and_view_importance()
            
            # Find most important tables
            top_tables = sorted(
                importance_data['tables'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]
            
            # Find most important views
            top_views = sorted(
                importance_data['views'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            return {
                'total_tables': len([t for t in self.tables.values() if not t.is_view]),
                'total_views': len([t for t in self.tables.values() if t.is_view]),
                'total_entities': len(self.tables),
                'graph_statistics': stats,
                'table_clusters': [list(cluster) for cluster in clusters],
                'most_important_tables': [
                    {'table': table, 'importance_score': score} 
                    for table, score in top_tables
                ],
                'most_important_views': [
                    {'table': table, 'importance_score': score} 
                    for table, score in top_views
                ],
                'database_type': self.extractor._get_db_type()
            }
        else:
            # Fallback to original behavior for NetworkX backend
            importance = self.backend.get_table_importance()
            
            # Find most important tables
            top_tables = sorted(
                importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]
            
            return {
                'total_tables': len(self.tables),
                'graph_statistics': stats,
                'table_clusters': [list(cluster) for cluster in clusters],
                'most_important_tables': [
                    {'table': table, 'importance_score': score} 
                    for table, score in top_tables
                ],
                'database_type': self.extractor._get_db_type()
            }
    
    def export_schema_subset(self, table_names: List[str]) -> Dict[str, Any]:
        """Export schema information for a subset of tables."""
        if not self._connected:
            raise RuntimeError("Schema not analyzed. Call analyze_schema() first.")
        
        subset = {}
        for table_name in table_names:
            if table_name in self.tables:
                table_info = self.tables[table_name]
                subset[table_name] = {
                    'columns': [
                        {
                            'name': col.name,
                            'type': col.type,
                            'nullable': col.nullable,
                            'primary_key': col.primary_key,
                            'foreign_key': col.foreign_key
                        }
                        for col in table_info.columns
                    ],
                    'foreign_keys': table_info.foreign_keys,
                    'relationships': list(self.backend.get_table_neighbors(table_name))
                }
        
        return subset
    
    def close(self) -> None:
        """Close database connection."""
        if self.extractor:
            self.extractor.close()