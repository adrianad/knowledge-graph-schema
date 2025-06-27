"""Schema analysis and table relationship discovery."""

from typing import Dict, List, Set, Tuple, Any, Optional
import logging
from dataclasses import dataclass

from .database import DatabaseExtractor, TableInfo
from .backends.base import GraphBackend


@dataclass
class TableRelevanceScore:
    """Table relevance score for a query."""
    table_name: str
    score: float
    reasons: List[str]


class SchemaAnalyzer:
    """Analyze database schema and provide intelligent table suggestions."""
    
    def __init__(self, connection_string: str, backend: str = 'networkx', **backend_kwargs):
        """Initialize analyzer with database connection and specified backend."""
        self.extractor = DatabaseExtractor(connection_string)
        self.logger = logging.getLogger(__name__)
        self._connected = False
        self.tables: Dict[str, TableInfo] = {}
        
        # Initialize the appropriate backend
        if backend == 'networkx':
            from .backends.networkx_backend import NetworkXBackend
            self.backend = NetworkXBackend()
        elif backend == 'neo4j':
            from .backends.neo4j_backend import Neo4jBackend
            self.backend = Neo4jBackend(**backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        self.logger.info(f"Initialized SchemaAnalyzer with {backend} backend")
    
    def analyze_schema(self, include_views: bool = True) -> None:
        """Analyze database schema and build knowledge graph."""
        self.extractor.connect()
        self._connected = True
        
        # Extract schema information
        self.tables = self.extractor.extract_schema(include_views=include_views)
        
        # Build knowledge graph using backend
        self.backend.build_from_schema(self.tables)
        
        self.logger.info("Schema analysis completed")
    
    def find_relevant_tables(
        self, 
        keywords: List[str], 
        max_tables: int = 10
    ) -> List[TableRelevanceScore]:
        """Find tables relevant to given keywords."""
        if not self._connected:
            raise RuntimeError("Schema not analyzed. Call analyze_schema() first.")
        
        scores = {}
        
        for table_name in self.tables:
            score, reasons = self._calculate_table_relevance(table_name, keywords)
            if score > 0:
                scores[table_name] = TableRelevanceScore(
                    table_name=table_name,
                    score=score,
                    reasons=reasons
                )
        
        # Sort by score and return top results
        sorted_scores = sorted(scores.values(), key=lambda x: x.score, reverse=True)
        return sorted_scores[:max_tables]
    
    def _calculate_table_relevance(
        self, 
        table_name: str, 
        keywords: List[str]
    ) -> Tuple[float, List[str]]:
        """Calculate relevance score for a table given keywords."""
        score = 0.0
        reasons = []
        
        table_info = self.tables[table_name]
        
        # Check table name matches
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact table name match
            if keyword_lower == table_name.lower():
                score += 10.0
                reasons.append(f"Exact table name match: {keyword}")
                continue
            
            # Partial table name match
            if keyword_lower in table_name.lower():
                score += 5.0
                reasons.append(f"Partial table name match: {keyword}")
            
            # Column name matches
            for column in table_info.columns:
                if keyword_lower == column.name.lower():
                    score += 3.0
                    reasons.append(f"Exact column match: {column.name}")
                elif keyword_lower in column.name.lower():
                    score += 1.0
                    reasons.append(f"Partial column match: {column.name}")
        
        return score, reasons
    
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
        if not self._connected:
            raise RuntimeError("Schema not analyzed. Call analyze_schema() first.")
        
        suggestions = set()
        
        for table in base_tables:
            if table in self.backend.get_all_tables():
                # Get neighbors (direct relationships)
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
    
    def find_connection_path(self, table1: str, table2: str) -> Optional[List[str]]:
        """Find connection path between two tables."""
        if not self._connected:
            raise RuntimeError("Schema not analyzed. Call analyze_schema() first.")
        
        return self.backend.find_shortest_path(table1, table2)
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get summary of the database schema."""
        if not self._connected:
            raise RuntimeError("Schema not analyzed. Call analyze_schema() first.")
        
        stats = self.backend.get_graph_stats()
        clusters = self.backend.find_table_clusters()
        importance = self.backend.get_table_importance()
        
        # Find most important tables
        top_tables = sorted(
            importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
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
            if table_name in self.graph.tables:
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