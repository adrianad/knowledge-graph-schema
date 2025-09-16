"""SQL execution with safety validation and result truncation."""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError

logger = logging.getLogger(__name__)

# Allowed SQL statement types (read-only operations)
ALLOWED_STATEMENTS = {
    'SELECT', 'EXPLAIN', 'SHOW', 'DESCRIBE', 'DESC', 'WITH'
}

# Dangerous keywords that should be blocked
BLOCKED_KEYWORDS = {
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
    'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'BEGIN', 'START',
    'LOCK', 'UNLOCK', 'SET', 'RESET', 'CALL', 'EXECUTE', 'EXEC'
}

class SafeSqlExecutor:
    """Safe SQL executor with validation and result truncation."""
    
    def __init__(self, connection_string: str, max_tokens: int = 10000):
        """
        Initialize SQL executor.
        
        Args:
            connection_string: Database connection string
            max_tokens: Maximum tokens before truncating results (estimated as chars/4)
        """
        self.connection_string = connection_string
        self.max_tokens = max_tokens
        self.engine = None
        # Read timeout from environment variable (default: 5000ms)
        self.timeout_ms = int(os.getenv('SQL_STATEMENT_TIMEOUT_MS', '5000'))
    
    def _connect(self):
        """Create database connection if not exists."""
        if not self.engine:
            try:
                # For PostgreSQL, add statement_timeout to connection parameters
                if 'postgresql' in self.connection_string.lower():
                    # Parse connection string and add timeout parameter
                    connect_args = {
                        'options': f'-c statement_timeout={self.timeout_ms}ms'
                    }
                    self.engine = create_engine(self.connection_string, connect_args=connect_args)
                else:
                    # For other databases, create engine without timeout setting
                    self.engine = create_engine(self.connection_string)
                logger.info(f"Database connection established with timeout: {self.timeout_ms}ms")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise
    
    def _validate_sql(self, sql: str) -> None:
        """
        Validate SQL for safety (read-only operations only).
        
        Args:
            sql: SQL query to validate
            
        Raises:
            ValueError: If SQL contains dangerous operations
        """
        # Clean up SQL - remove comments and extra whitespace
        cleaned_sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)  # Remove /* */ comments
        cleaned_sql = re.sub(r'--.*$', '', cleaned_sql, flags=re.MULTILINE)  # Remove -- comments
        cleaned_sql = cleaned_sql.strip().upper()
        
        if not cleaned_sql:
            raise ValueError("Empty SQL query")
        
        # Check for blocked keywords
        for keyword in BLOCKED_KEYWORDS:
            if re.search(r'\b' + keyword + r'\b', cleaned_sql):
                raise ValueError(f"Blocked keyword detected: {keyword}")
        
        # Check if it starts with an allowed statement
        first_word = cleaned_sql.split()[0] if cleaned_sql.split() else ''
        if first_word not in ALLOWED_STATEMENTS:
            raise ValueError(f"Only read-only operations are allowed. Got: {first_word}")
        
        # Additional safety checks
        if ';' in cleaned_sql:
            # Check if there are multiple statements
            statements = [s.strip() for s in cleaned_sql.split(';') if s.strip()]
            if len(statements) > 1:
                raise ValueError("Multiple statements not allowed")
        
        logger.info("SQL validation passed")
    
    def _format_results(self, rows: List[Any], columns: List[str]) -> str:
        """
        Format query results as text with truncation.
        
        Args:
            rows: Query result rows
            columns: Column names
            
        Returns:
            Formatted text results
        """
        if not rows:
            return "No results returned"
        
        # Start building result text
        result_lines = []
        
        # Add column headers
        header = " | ".join(str(col) for col in columns)
        result_lines.append(header)
        result_lines.append("-" * len(header))
        
        # Add data rows
        total_chars = len(header) + len("-" * len(header))  # Count header chars
        truncated = False
        
        for row in rows:
            # Convert row to strings and handle None values
            row_values = []
            for value in row:
                if value is None:
                    row_values.append("NULL")
                elif isinstance(value, (int, float)):
                    row_values.append(str(value))
                else:
                    # Truncate very long text fields
                    str_value = str(value)
                    if len(str_value) > 1000:
                        str_value = str_value[:1000] + "..."
                    row_values.append(str_value)
            
            row_text = " | ".join(row_values)
            
            # Check token count (estimate: chars / 4)
            row_chars = len(row_text)
            estimated_tokens = (total_chars + row_chars) // 4
            
            if estimated_tokens > self.max_tokens:
                result_lines.append(f"... (truncated at {len(result_lines)-2} rows due to token limit)")
                truncated = True
                break
            
            result_lines.append(row_text)
            total_chars += row_chars
        
        result_text = "\n".join(result_lines)
        
        # Add summary
        actual_rows = len(result_lines) - 2  # Subtract header and separator
        if truncated:
            result_text += f"\n\nShowing {actual_rows} rows (truncated)"
        else:
            result_text += f"\n\n{actual_rows} row(s) returned"
        
        return result_text
    
    def execute_query(self, sql: str) -> str:
        """
        Execute SQL query safely with validation and truncation.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Formatted query results or error message
        """
        try:
            # Validate SQL first
            self._validate_sql(sql)
            
            # Connect to database
            self._connect()
            
            # Execute query in read-only transaction
            with self.engine.connect() as conn:
                # Start read-only transaction for additional safety
                with conn.begin() as trans:
                    try:
                        # Execute query
                        result = conn.execute(text(sql))
                        
                        # Fetch results
                        if result.returns_rows:
                            rows = result.fetchall()
                            columns = list(result.keys())
                            return self._format_results(rows, columns)
                        else:
                            # For queries that don't return rows (like EXPLAIN)
                            return "Query executed successfully (no rows returned)"
                            
                    except Exception as e:
                        trans.rollback()
                        raise e
                        
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            # Check if this is a timeout error
            error_str = str(e).lower()
            if 'timeout' in error_str or 'canceling statement due to statement timeout' in error_str:
                return "Query takes too long and was aborted."
            return f"Database error: {str(e)}"
        except ValueError as e:
            logger.error(f"SQL validation error: {e}")
            return f"SQL validation error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            # Check if this is a timeout error from other sources
            error_str = str(e).lower()
            if 'timeout' in error_str:
                return "Query takes too long and was aborted."
            return f"Error: {str(e)}"
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


def create_sql_executor(connection_string: Optional[str] = None, max_tokens: int = 10000) -> SafeSqlExecutor:
    """
    Create a SQL executor instance.
    
    Args:
        connection_string: Database connection string (uses DATABASE_URL env var if not provided)
        max_tokens: Maximum tokens before truncation
        
    Returns:
        SafeSqlExecutor instance
    """
    if not connection_string:
        connection_string = os.getenv('DATABASE_URL')
        if not connection_string:
            raise ValueError("No database connection string provided and DATABASE_URL not set")
    
    return SafeSqlExecutor(connection_string, max_tokens)