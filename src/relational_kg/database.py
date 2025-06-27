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
            for view_name in inspector.get_view_names():
                columns = self._extract_columns(inspector, view_name, is_view=True)
                view_definition = self._get_view_definition(view_name)
                view_dependencies = self._parse_view_dependencies(view_definition, tables.keys())
                
                tables[view_name] = TableInfo(
                    name=view_name,
                    columns=columns,
                    foreign_keys=[],  # Views don't have foreign keys themselves
                    is_view=True,
                    view_definition=view_definition,
                    view_dependencies=view_dependencies
                )
            
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
        
        # Look for FROM and JOIN clauses
        # This is a simplified parser - more sophisticated parsing might be needed for complex views
        patterns = [
            r'\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\binner\s+join\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bleft\s+join\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bright\s+join\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bfull\s+join\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, view_def_lower)
            for match in matches:
                # Check if the match is actually a table/view name we know about
                if match in [t.lower() for t in available_tables]:
                    # Find the original case version
                    for table in available_tables:
                        if table.lower() == match:
                            if table not in dependencies:
                                dependencies.append(table)
                            break
        
        return dependencies
    
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