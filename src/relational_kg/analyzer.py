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
        max_suggestions: int = 5,
        max_hops: int = 1
    ) -> Dict[str, List[str]]:
        """Suggest additional tables that could be joined with base tables.
        
        Returns:
            Dict mapping each base table to its top suggestions
        """
        # Neo4j backend can work directly without schema analysis
        base_table_suggestions = {}
        
        # Use batch query for efficiency when available
        if hasattr(self.backend, 'get_table_neighbors_batch'):
            neighbor_results = self.backend.get_table_neighbors_batch(base_tables, max_hops)
        else:
            # Fallback to individual queries
            neighbor_results = {}
            for table in base_tables:
                neighbor_results[table] = self.backend.get_table_neighbors(table, max_hops)
        
        # Get importance scores once
        importance_scores = self.backend.get_table_importance()
        
        # Generate suggestions per base table
        for base_table in base_tables:
            neighbors = neighbor_results.get(base_table, set())
            # Remove other base tables from suggestions
            table_suggestions = neighbors - set(base_tables)
            
            # Sort by importance and take top suggestions per table
            sorted_suggestions = sorted(
                table_suggestions, 
                key=lambda x: importance_scores.get(x, 0), 
                reverse=True
            )
            
            base_table_suggestions[base_table] = sorted_suggestions[:max_suggestions]
        
        return base_table_suggestions
    
    def suggest_tables_for_join_combined(
        self, 
        base_tables: List[str], 
        max_suggestions: int = 5,
        max_hops: int = 1
    ) -> List[str]:
        """Suggest additional tables (combined results) that could be joined with base tables."""
        per_table_suggestions = self.suggest_tables_for_join(base_tables, max_suggestions, max_hops)
        
        # Combine all suggestions and deduplicate
        all_suggestions = set()
        for suggestions in per_table_suggestions.values():
            all_suggestions.update(suggestions)
        
        # Score and sort combined suggestions
        importance_scores = self.backend.get_table_importance()
        sorted_suggestions = sorted(
            all_suggestions, 
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