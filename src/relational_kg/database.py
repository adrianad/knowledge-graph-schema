"""Database connection and schema extraction."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import logging
import re

from sqlalchemy import create_engine, inspect, MetaData, Table, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


@dataclass
class ColumnInfo:
    """Column information."""
    name: str
    type: str
    nullable: bool
    primary_key: bool
    foreign_key: Optional[str] = None
    

@dataclass
class TableInfo:
    """Table information."""
    name: str
    columns: List[ColumnInfo]
    foreign_keys: List[Dict[str, Any]]
    is_view: bool = False
    view_definition: Optional[str] = None
    view_dependencies: List[str] = None  # Tables/views this view depends on
    
    def __post_init__(self):
        if self.view_dependencies is None:
            self.view_dependencies = []


class DatabaseExtractor:
    """Extract schema information from databases."""
    
    def __init__(self, connection_string: str):
        """Initialize with database connection string."""
        self.connection_string = connection_string
        self.engine: Optional[Engine] = None
        self.metadata: Optional[MetaData] = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> None:
        """Establish database connection."""
        try:
            self.engine = create_engine(self.connection_string)
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            self.logger.info(f"Connected to database: {self._get_db_type()}")
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _get_db_type(self) -> str:
        """Get database type from connection string."""
        parsed = urlparse(self.connection_string)
        return parsed.scheme.split('+')[0]
    
    def extract_schema(self, include_views: bool = True) -> Dict[str, TableInfo]:
        """Extract complete schema information including views."""
        if not self.engine or not self.metadata:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        inspector = inspect(self.engine)
        tables = {}
        
        # Extract regular tables
        for table_name in inspector.get_table_names():
            columns = self._extract_columns(inspector, table_name)
            foreign_keys = self._extract_foreign_keys(inspector, table_name)
            
            tables[table_name] = TableInfo(
                name=table_name,
                columns=columns,
                foreign_keys=foreign_keys,
                is_view=False
            )
        
        # Extract views if requested
        if include_views:
            # First pass: collect all views without dependencies
            view_definitions = {}
            
            # Regular views
            for view_name in inspector.get_view_names():
                columns = self._extract_columns(inspector, view_name, is_view=True)
                view_definition = self._get_view_definition(view_name)
                view_definitions[view_name] = view_definition
                
                tables[view_name] = TableInfo(
                    name=view_name,
                    columns=columns,
                    foreign_keys=[],  # Views don't have foreign keys themselves
                    is_view=True,
                    view_definition=view_definition,
                    view_dependencies=[]  # Will be filled in second pass
                )
            
            # Materialized views (PostgreSQL)
            materialized_views = self._get_materialized_view_names()
            for view_name in materialized_views:
                columns = self._extract_columns(inspector, view_name, is_view=True)
                view_definition = self._get_materialized_view_definition(view_name)
                view_definitions[view_name] = view_definition
                
                tables[view_name] = TableInfo(
                    name=view_name,
                    columns=columns,
                    foreign_keys=[],  # Views don't have foreign keys themselves
                    is_view=True,
                    view_definition=view_definition,
                    view_dependencies=[]  # Will be filled in second pass
                )
            
            # Second pass: parse dependencies now that all tables/views are collected
            for view_name, view_definition in view_definitions.items():
                view_dependencies = self._parse_view_dependencies(view_definition, list(tables.keys()))
                tables[view_name].view_dependencies = view_dependencies
            
        self.logger.info(f"Extracted {len([t for t in tables.values() if not t.is_view])} tables and {len([t for t in tables.values() if t.is_view])} views from schema")
        return tables
    
    def _extract_columns(self, inspector, table_name: str, is_view: bool = False) -> List[ColumnInfo]:
        """Extract column information for a table or view."""
        columns = []
        column_info = inspector.get_columns(table_name)
        
        if not is_view:
            pk_constraint = inspector.get_pk_constraint(table_name)
            fk_constraints = inspector.get_foreign_keys(table_name)
            
            primary_keys = pk_constraint.get('constrained_columns', [])
            
            # Build foreign key mapping
            fk_map = {}
            for fk in fk_constraints:
                for local_col, ref_col in zip(fk['constrained_columns'], fk['referred_columns']):
                    fk_map[local_col] = f"{fk['referred_table']}.{ref_col}"
        else:
            primary_keys = []
            fk_map = {}
        
        for col in column_info:
            columns.append(ColumnInfo(
                name=col['name'],
                type=str(col['type']),
                nullable=col['nullable'],
                primary_key=col['name'] in primary_keys,
                foreign_key=fk_map.get(col['name'])
            ))
        
        return columns
    
    def _extract_foreign_keys(self, inspector, table_name: str) -> List[Dict[str, Any]]:
        """Extract foreign key relationships for a table."""
        foreign_keys = []
        fk_constraints = inspector.get_foreign_keys(table_name)
        
        for fk in fk_constraints:
            foreign_keys.append({
                'name': fk.get('name'),
                'constrained_columns': fk['constrained_columns'],
                'referred_table': fk['referred_table'],
                'referred_columns': fk['referred_columns'],
                'ondelete': fk.get('ondelete'),
                'onupdate': fk.get('onupdate')
            })
        
        return foreign_keys
    
    def _get_view_definition(self, view_name: str) -> Optional[str]:
        """Get view definition SQL."""
        if not self.engine:
            return None
        
        db_type = self._get_db_type()
        
        try:
            with self.engine.connect() as conn:
                if db_type == 'postgresql':
                    result = conn.execute(text("""
                        SELECT definition 
                        FROM pg_views 
                        WHERE viewname = :view_name
                    """), {"view_name": view_name})
                elif db_type == 'mysql':
                    result = conn.execute(text("""
                        SELECT VIEW_DEFINITION 
                        FROM information_schema.VIEWS 
                        WHERE TABLE_NAME = :view_name
                    """), {"view_name": view_name})
                elif db_type == 'sqlite':
                    result = conn.execute(text("""
                        SELECT sql 
                        FROM sqlite_master 
                        WHERE type = 'view' AND name = :view_name
                    """), {"view_name": view_name})
                else:
                    self.logger.warning(f"View definition extraction not implemented for {db_type}")
                    return None
                
                row = result.fetchone()
                return row[0] if row else None
                
        except SQLAlchemyError as e:
            self.logger.warning(f"Failed to get view definition for {view_name}: {e}")
            return None
    
    def _parse_view_dependencies(self, view_definition: Optional[str], available_tables: List[str]) -> List[str]:
        """Parse view definition to find table/view dependencies."""
        if not view_definition:
            return []
        
        dependencies = []
        
        # Convert to lowercase for case-insensitive matching
        view_def_lower = view_definition.lower()
        
        # Remove extra whitespace and newlines for easier parsing
        view_def_clean = re.sub(r'\s+', ' ', view_def_lower.strip())
        
        # More comprehensive patterns to handle complex SQL
        patterns = [
            # FROM clauses - simple table references
            r'\bfrom\s+(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)',
            # FROM clauses with parentheses
            r'\bfrom\s+\(\s*(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)',
            # All JOIN variations
            r'\b(?:inner\s+|left\s+|right\s+|full\s+|cross\s+)?(?:outer\s+)?join\s+(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)',
            # Tables in parenthetical expressions like "(public.table1 alias"
            r'\(\s*(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s+[a-zA-Z_][a-zA-Z0-9_]*',
        ]
        
        # Apply patterns
        for pattern in patterns:
            matches = re.findall(pattern, view_def_clean)
            for match in matches:
                self._add_dependency_if_exists(match, available_tables, dependencies)
        
        # Special handling for complex FROM clauses with aliases
        # Pattern: "FROM (public.table1 alias JOIN public.table2 alias2 ON ...)"
        from_clause_pattern = r'\bfrom\s+\((.*?)\)'
        from_match = re.search(from_clause_pattern, view_def_clean, re.DOTALL)
        if from_match:
            from_content = from_match.group(1)
            # Extract all table names from the FROM clause content
            table_pattern = r'(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s+[a-zA-Z_][a-zA-Z0-9_]*'
            table_matches = re.findall(table_pattern, from_content)
            for match in table_matches:
                self._add_dependency_if_exists(match, available_tables, dependencies)
        
        # Handle WHERE clause table references
        where_patterns = [
            r'where\s+\([^)]*\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=',
            r'and\s+\([^)]*\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=',
        ]
        
        for pattern in where_patterns:
            matches = re.findall(pattern, view_def_clean)
            for match in matches:
                self._add_dependency_if_exists(match, available_tables, dependencies)
        
        # Log what we found for debugging
        if dependencies:
            self.logger.info(f"Found dependencies for view: {dependencies}")
        else:
            self.logger.warning(f"No dependencies found in view definition: {view_definition[:200]}...")
        
        return dependencies
    
    def _add_dependency_if_exists(self, table_candidate: str, available_tables: List[str], dependencies: List[str]) -> None:
        """Add table to dependencies if it exists in available tables."""
        if table_candidate in [t.lower() for t in available_tables]:
            # Find the original case version
            for table in available_tables:
                if table.lower() == table_candidate:
                    if table not in dependencies:
                        dependencies.append(table)
                    break
    
    def _get_materialized_view_names(self) -> List[str]:
        """Get list of materialized view names (PostgreSQL specific)."""
        if not self.engine:
            return []
        
        db_type = self._get_db_type()
        
        if db_type != 'postgresql':
            return []  # Only PostgreSQL supports materialized views
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT schemaname, matviewname 
                    FROM pg_matviews 
                    WHERE schemaname = 'public'
                """))
                
                return [row[1] for row in result]
                
        except SQLAlchemyError as e:
            self.logger.warning(f"Failed to get materialized view names: {e}")
            return []
    
    def _get_materialized_view_definition(self, view_name: str) -> Optional[str]:
        """Get materialized view definition SQL (PostgreSQL specific)."""
        if not self.engine:
            return None
        
        db_type = self._get_db_type()
        
        if db_type != 'postgresql':
            return None
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT definition 
                    FROM pg_matviews 
                    WHERE matviewname = :view_name AND schemaname = 'public'
                """), {"view_name": view_name})
                
                row = result.fetchone()
                return row[0] if row else None
                
        except SQLAlchemyError as e:
            self.logger.warning(f"Failed to get materialized view definition for {view_name}: {e}")
            return None
    
    def get_table_names(self) -> List[str]:
        """Get list of table names."""
        if not self.metadata:
            raise RuntimeError("Database not connected. Call connect() first.")
        return list(self.metadata.tables.keys())
    
    def get_view_names(self) -> List[str]:
        """Get list of view names."""
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        inspector = inspect(self.engine)
        return inspector.get_view_names()
    
    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")