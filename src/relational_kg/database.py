"""Database connection and schema extraction."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import logging

from sqlalchemy import create_engine, inspect, MetaData, Table
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
    
    def extract_schema(self) -> Dict[str, TableInfo]:
        """Extract complete schema information."""
        if not self.engine or not self.metadata:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        inspector = inspect(self.engine)
        tables = {}
        
        for table_name in inspector.get_table_names():
            columns = self._extract_columns(inspector, table_name)
            foreign_keys = self._extract_foreign_keys(inspector, table_name)
            
            tables[table_name] = TableInfo(
                name=table_name,
                columns=columns,
                foreign_keys=foreign_keys
            )
            
        self.logger.info(f"Extracted {len(tables)} tables from schema")
        return tables
    
    def _extract_columns(self, inspector, table_name: str) -> List[ColumnInfo]:
        """Extract column information for a table."""
        columns = []
        column_info = inspector.get_columns(table_name)
        pk_constraint = inspector.get_pk_constraint(table_name)
        fk_constraints = inspector.get_foreign_keys(table_name)
        
        primary_keys = pk_constraint.get('constrained_columns', [])
        
        # Build foreign key mapping
        fk_map = {}
        for fk in fk_constraints:
            for local_col, ref_col in zip(fk['constrained_columns'], fk['referred_columns']):
                fk_map[local_col] = f"{fk['referred_table']}.{ref_col}"
        
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
    
    def get_table_names(self) -> List[str]:
        """Get list of table names."""
        if not self.metadata:
            raise RuntimeError("Database not connected. Call connect() first.")
        return list(self.metadata.tables.keys())
    
    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")