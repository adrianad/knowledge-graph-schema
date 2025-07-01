"""DDL generation utilities for converting TableInfo objects to SQL DDL statements."""

from typing import List, Dict, Any, Optional
from .database import TableInfo, ColumnInfo


def generate_table_ddl(table_info: TableInfo, dialect: str = 'postgresql') -> str:
    """Generate CREATE TABLE or CREATE VIEW DDL statement from TableInfo.
    
    Args:
        table_info: Table information containing columns, constraints, etc.
        dialect: SQL dialect (postgresql, mysql, sqlite)
        
    Returns:
        DDL statement as string
    """
    if table_info.is_view:
        return _generate_view_ddl(table_info, dialect)
    else:
        return _generate_table_ddl(table_info, dialect)


def _generate_table_ddl(table_info: TableInfo, dialect: str) -> str:
    """Generate CREATE TABLE statement."""
    lines = [f"CREATE TABLE {table_info.name} ("]
    
    # Column definitions
    column_lines = []
    for col in table_info.columns:
        col_def = f"    {col.name} {_format_column_type(col.type, dialect)}"
        
        # Add NOT NULL constraint
        if not col.nullable:
            col_def += " NOT NULL"
        
        # Add PRIMARY KEY constraint (inline for single column)
        if col.primary_key:
            primary_keys = [c.name for c in table_info.columns if c.primary_key]
            if len(primary_keys) == 1:
                col_def += " PRIMARY KEY"
        
        column_lines.append(col_def)
    
    lines.extend(column_lines)
    
    # Add composite primary key constraint if multiple columns
    primary_keys = [col.name for col in table_info.columns if col.primary_key]
    if len(primary_keys) > 1:
        pk_constraint = f"    PRIMARY KEY ({', '.join(primary_keys)})"
        lines.append(pk_constraint)
    
    # Add foreign key constraints
    for fk in table_info.foreign_keys:
        if fk.get('constrained_columns') and fk.get('referred_table'):
            constrained_cols = ', '.join(fk['constrained_columns'])
            referred_table = fk['referred_table']
            
            # For foreign keys, we'll use a simple REFERENCES clause
            # Most databases support this format
            fk_constraint = f"    FOREIGN KEY ({constrained_cols}) REFERENCES {referred_table}"
            if fk.get('referred_columns'):
                referred_cols = ', '.join(fk['referred_columns'])
                fk_constraint += f"({referred_cols})"
            
            lines.append(fk_constraint)
    
    lines.append(");")
    return '\n'.join(lines)


def _generate_view_ddl(table_info: TableInfo, dialect: str) -> str:
    """Generate CREATE VIEW statement."""
    if table_info.view_definition:
        # Use original view definition if available
        return f"CREATE VIEW {table_info.name} AS\n{table_info.view_definition};"
    else:
        # Fallback to column list if no definition available
        columns = [col.name for col in table_info.columns]
        return f"CREATE VIEW {table_info.name} (\n    {', '.join(columns)}\n) AS\n-- View definition not available;"


def _format_column_type(column_type: str, dialect: str) -> str:
    """Format column type for specific SQL dialect.
    
    Args:
        column_type: SQLAlchemy column type as string
        dialect: Target SQL dialect
        
    Returns:
        Formatted column type string
    """
    # Convert common SQLAlchemy types to dialect-specific types
    type_str = str(column_type).upper()
    
    # Handle common type mappings
    type_mappings = {
        'postgresql': {
            'INTEGER': 'INTEGER',
            'VARCHAR': 'VARCHAR',
            'TEXT': 'TEXT',
            'BOOLEAN': 'BOOLEAN',
            'TIMESTAMP': 'TIMESTAMP',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'DECIMAL': 'DECIMAL',
            'NUMERIC': 'NUMERIC',
            'FLOAT': 'REAL',
            'DOUBLE': 'DOUBLE PRECISION',
            'BIGINT': 'BIGINT',
            'SMALLINT': 'SMALLINT'
        },
        'mysql': {
            'INTEGER': 'INT',
            'VARCHAR': 'VARCHAR',
            'TEXT': 'TEXT',
            'BOOLEAN': 'BOOLEAN',
            'TIMESTAMP': 'TIMESTAMP',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'DECIMAL': 'DECIMAL',
            'NUMERIC': 'DECIMAL',
            'FLOAT': 'FLOAT',
            'DOUBLE': 'DOUBLE',
            'BIGINT': 'BIGINT',
            'SMALLINT': 'SMALLINT'
        },
        'sqlite': {
            'INTEGER': 'INTEGER',
            'VARCHAR': 'TEXT',
            'TEXT': 'TEXT',
            'BOOLEAN': 'INTEGER',
            'TIMESTAMP': 'TEXT',
            'DATE': 'TEXT',
            'TIME': 'TEXT',
            'DECIMAL': 'REAL',
            'NUMERIC': 'REAL',
            'FLOAT': 'REAL',
            'DOUBLE': 'REAL',
            'BIGINT': 'INTEGER',
            'SMALLINT': 'INTEGER'
        }
    }
    
    # Get dialect-specific mappings
    mappings = type_mappings.get(dialect, type_mappings['postgresql'])
    
    # Handle parameterized types like VARCHAR(255)
    if '(' in type_str:
        base_type = type_str.split('(')[0]
        params = type_str.split('(')[1].rstrip(')')
        
        if base_type in mappings:
            return f"{mappings[base_type]}({params})"
        else:
            return type_str
    
    # Handle simple types
    if type_str in mappings:
        return mappings[type_str]
    
    # Return original type if no mapping found
    return type_str


def generate_schema_ddl(tables: Dict[str, TableInfo], dialect: str = 'postgresql') -> str:
    """Generate DDL for entire schema.
    
    Args:
        tables: Dictionary of table name to TableInfo
        dialect: SQL dialect
        
    Returns:
        Complete DDL as string
    """
    ddl_statements = []
    
    # Generate table DDL first (views may depend on tables)
    table_ddl = []
    view_ddl = []
    
    for table_name, table_info in tables.items():
        ddl = generate_table_ddl(table_info, dialect)
        if table_info.is_view:
            view_ddl.append(ddl)
        else:
            table_ddl.append(ddl)
    
    # Combine tables first, then views
    ddl_statements.extend(table_ddl)
    ddl_statements.extend(view_ddl)
    
    return '\n\n'.join(ddl_statements)