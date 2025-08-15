"""Model Context Protocol (MCP) server for relational knowledge graph interactions."""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import Field

from mcp.server.fastmcp import FastMCP
MCP_AVAILABLE = True

from .analyzer import SchemaAnalyzer
from .sql_executor import create_sql_executor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Knowledge Graph Server")

def _get_analyzer() -> SchemaAnalyzer:
    """Create and return a Neo4j analyzer instance."""
    # Get Neo4j connection details from environment
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD environment variable is required")
    
    # Use dummy connection string since we only need Neo4j backend
    analyzer = SchemaAnalyzer(
        connection_string="dummy://localhost",
        backend='neo4j',
        uri=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password
    )
    
    return analyzer


@mcp.tool(description="Get detailed information about specific tables from Neo4j graph")
def explore_table(
    table_names: str = Field(description="Comma-separated list of table names to explore")
) -> str:
    """Get detailed information about specific tables from Neo4j graph.
    
    Args:
        table_names: Comma-separated list of table names to explore
        
    Returns:
        Plain text DDL for the requested tables
    """
    try:
        analyzer = _get_analyzer()
        
        # Parse table names (handle comma-separated input)
        table_list = []
        for name in table_names.split(','):
            name = name.strip()
            if name:
                table_list.append(name)
        
        if not table_list:
            return "Error: No table names provided"
        
        logger.info(f"Getting DDL for tables: {table_list}")
        
        # Get table DDL from Neo4j
        tables_ddl = analyzer.backend.get_table_ddl(table_list)
        
        # Separate found and missing tables  
        found_tables = [t for t in tables_ddl if not t.get('not_found', False)]
        missing_tables = [t['name'] for t in tables_ddl if t.get('not_found', False)]
        
        # Format DDL output - one table per line, no newlines within DDL
        ddl_statements = []
        for table in found_tables:
            # Replace newlines within DDL with spaces
            clean_ddl = table['ddl'].replace('\n', ' ')
            ddl_statements.append(f"-- {table['name']} {clean_ddl}")
        
        result_text = "\n".join(ddl_statements)
        
        if missing_tables:
            if result_text:
                result_text += f"\n\nWarning: Tables not found in Neo4j: {', '.join(missing_tables)}"
            else:
                result_text = f"Error: Tables not found in Neo4j: {', '.join(missing_tables)}\nMake sure Neo4j is running and the knowledge graph has been built with 'rkg analyze'"
        
        analyzer.close()
        return result_text
        
    except Exception as e:
        logger.error(f"Error in explore_table: {e}")
        return f"Error: {str(e)}\nMake sure Neo4j is running and the knowledge graph has been built with 'rkg analyze'"


@mcp.tool(description="List all available table clusters from Neo4j")
def list_clusters(
    exclude_main: bool = Field(default=True, description="Whether to exclude the clusters that make up the main cluster")
) -> str:
    """List all available table clusters from Neo4j.
    
    Args:
        exclude_main: Whether to exclude the clusters that make up the main cluster
    
    Returns:
        Dictionary containing all clusters with their basic information
    """
    try:
        analyzer = _get_analyzer()
        
        logger.info("Retrieving all clusters from Neo4j")
        
        # Get all clusters from Neo4j
        clusters = analyzer.backend.get_all_clusters()
        
        # Filter out main clusters if exclude_main is True
        if exclude_main:
            # Get main cluster size from environment
            main_cluster_size = int(os.getenv('MAIN_CLUSTER_SIZE', '2'))
            
            if clusters and len(clusters) >= main_cluster_size:
                # Get main cluster tables for filtering
                main_cluster_tables = set()
                for i in range(main_cluster_size):
                    main_tables = clusters[i].get('tables', [])
                    main_cluster_tables.update(main_tables)
                
                # Skip the first N clusters (they make up the main cluster)
                remaining_clusters = clusters[main_cluster_size:]
                
                # Filter main cluster tables from remaining clusters
                for cluster in remaining_clusters:
                    cluster_tables = cluster.get('tables', [])
                    filtered_tables = [table for table in cluster_tables 
                                     if table not in main_cluster_tables]
                    cluster['tables'] = filtered_tables
                
                clusters = remaining_clusters
        
        if not clusters:
            return "No clusters found. Run 'rkg create-clusters' to generate clusters first."
        
        # Format clusters as simple text - just cluster IDs
        cluster_ids = [cluster.get('id', 'unknown') for cluster in clusters]
        analyzer.close()
        return ", ".join(cluster_ids)
        
    except Exception as e:
        logger.error(f"Error in list_clusters: {e}")
        return f"Error: {str(e)}. Make sure Neo4j is running and clusters have been created with 'rkg create-clusters'"


@mcp.tool(description="Show detailed information about a specific cluster")
def show_cluster(
    cluster_id: str = Field(description="The cluster ID to show details for"),
    detailed: bool = Field(default=False, description="Whether to include detailed column information for tables"),
    exclude_main: bool = Field(default=True, description="Whether to exclude tables that are in the main cluster")
) -> str:
    """Show detailed information about a specific cluster.
    
    Args:
        cluster_id: The cluster ID to show details for
        detailed: Whether to include detailed column information for tables
        exclude_main: Whether to exclude tables that are in the main cluster
        
    Returns:
        Dictionary containing detailed cluster information and its tables
    """
    try:
        analyzer = _get_analyzer()
        
        if not cluster_id:
            return {"error": "cluster_id is required"}
        
        logger.info(f"Getting DDL for cluster: {cluster_id}")
        
        # Get cluster metadata and table list
        all_clusters = analyzer.backend.get_all_clusters()
        cluster_info = None
        for cluster in all_clusters:
            if cluster['id'] == cluster_id:
                cluster_info = cluster
                break
        
        if not cluster_info:
            return f"Error: Cluster '{cluster_id}' not found. Use list_clusters to see available cluster IDs"
        
        # Get table names from cluster
        cluster_tables = cluster_info.get('tables', [])
        
        # Filter out main cluster tables if exclude_main is True
        if exclude_main:
            # Get main cluster size from environment
            main_cluster_size = int(os.getenv('MAIN_CLUSTER_SIZE', '2'))
            
            # Get main cluster tables
            main_cluster_tables = set()
            if all_clusters and len(all_clusters) >= main_cluster_size:
                for i in range(main_cluster_size):
                    main_tables = all_clusters[i].get('tables', [])
                    main_cluster_tables.update(main_tables)
            
            # Filter out main cluster tables
            cluster_tables = [table for table in cluster_tables 
                            if table not in main_cluster_tables]
        
        if detailed:
            # Get cluster table DDL from Neo4j
            tables_ddl = analyzer.backend.get_cluster_tables_ddl(cluster_id)
            
            # Filter DDL results to match filtered table list
            filtered_ddl = [table for table in tables_ddl 
                          if table['name'] in cluster_tables]
            
            # Return just DDL - one table per line, no newlines within DDL
            ddl_statements = []
            for table in filtered_ddl:
                clean_ddl = table['ddl'].replace('\n', ' ')
                ddl_statements.append(f"-- {table['name']} {clean_ddl}")
            
            result_text = "\n".join(ddl_statements)
        else:
            # Return comma-separated table names
            result_text = ", ".join(cluster_tables)
        
        analyzer.close()
        return result_text
        
    except Exception as e:
        logger.error(f"Error in show_cluster: {e}")
        return f"Error: {str(e)}. Make sure Neo4j is running and the cluster exists"


@mcp.tool(description="Get the main cluster (union of top N most important clusters) without duplicates")
def get_main_cluster(
    detailed: bool = Field(default=False, description="Whether to include detailed DDL information")
) -> str:
    """Get the main cluster (union of top N most important clusters) without duplicates.
    
    Args:
        detailed: Whether to include detailed DDL information
        
    Returns:
        Dictionary containing main cluster information and tables
    """
    try:
        analyzer = _get_analyzer()
        
        # Get main cluster size from environment
        main_cluster_size = int(os.getenv('MAIN_CLUSTER_SIZE', '2'))
        
        logger.info(f"Getting main cluster from top {main_cluster_size} clusters")
        
        # Get all clusters ordered by importance (first clusters are most important)
        clusters = analyzer.backend.get_all_clusters()
        
        if not clusters:
            return "Error: No clusters found in the database. Use 'rkg create-clusters' to generate clusters first"
        
        if len(clusters) < main_cluster_size:
            logger.warning(f"Only {len(clusters)} clusters available, using all of them")
            main_cluster_size = len(clusters)
        
        # Get top clusters and combine their tables
        main_cluster_tables = set()
        used_clusters = []
        
        for i in range(main_cluster_size):
            cluster = clusters[i]
            cluster_tables = cluster.get('tables', [])
            main_cluster_tables.update(cluster_tables)
            used_clusters.append({
                'id': cluster['id'],
                'name': cluster['name'],
                'table_count': len(cluster_tables)
            })
        
        # Convert to sorted list for consistent output
        main_cluster_tables = sorted(list(main_cluster_tables))
        
        if detailed:
            # Get DDL for all tables
            tables_ddl = analyzer.backend.get_table_ddl(main_cluster_tables)
            
            # Format DDL output - one table per line, no newlines within DDL
            ddl_statements = []
            for table in tables_ddl:
                if not table.get('not_found', False):
                    clean_ddl = table['ddl'].replace('\n', ' ')
                    ddl_statements.append(f"-- {table['name']} {clean_ddl}")
            
            result_text = "\n".join(ddl_statements)
        else:
            # Return comma-separated table names
            result_text = ", ".join(main_cluster_tables)
        
        analyzer.close()
        return result_text
        
    except Exception as e:
        logger.error(f"Error in get_main_cluster: {e}")
        return f"Error: {str(e)}. Make sure Neo4j is running and clusters have been created"


@mcp.tool(description="Find all connection paths between the given tables")
def find_path(
    tables: str = Field(description="Comma-separated list of tables to find connections between"),
    max_hops: int = Field(default=3, description="Maximum relationship hops to explore")
) -> str:
    """Find all connection paths between the given tables.
    
    Args:
        tables: Comma-separated list of tables to find connections between
        max_hops: Maximum relationship hops to explore (default: 3)
        
    Returns:
        Dictionary containing all connection paths organized by hop distance
    """
    try:
        analyzer = _get_analyzer()
        
        # Parse table names
        table_list = []
        for name in tables.split(','):
            name = name.strip()
            if name:
                table_list.append(name)
        
        if len(table_list) < 2:
            return "Error: At least 2 tables are required to find connections. Provide comma-separated table names like 'user_,sample,booking'"
        
        logger.info(f"Finding connections between tables: {table_list} (max_hops: {max_hops})")
        
        # Find all connections
        connections = analyzer.backend.find_all_connections(table_list, max_hops)
        
        if not connections:
            return f"No connections found between these tables within {max_hops} hops"
        
        # Format connections as simple path strings
        path_strings = []
        for conn in connections:
            path_str = " - ".join(conn['path'])
            path_strings.append(path_str)
        
        analyzer.close()
        return "\n".join(path_strings)
        
    except Exception as e:
        logger.error(f"Error in find_path: {e}")
        return f"Error: {str(e)}. Make sure Neo4j is running and the knowledge graph has been built"


@mcp.tool(description="Suggest additional tables that could be joined with the given base tables")
def suggest_joins(
    base_tables: str = Field(description="Comma-separated list of base tables to suggest joins for"),
    max_suggestions: int = Field(default=5, description="Maximum number of suggestions to return"),
    max_hops: int = Field(default=1, description="Maximum relationship hops to explore (1=direct, 2=two-hop, etc.)"),
    per_table: bool = Field(default=False, description="If True, return suggestions organized per base table; if False, return combined results")
) -> str:
    """Suggest additional tables that could be joined with the given base tables.
    
    Args:
        base_tables: Comma-separated list of base tables to suggest joins for
        max_suggestions: Maximum number of suggestions to return
        max_hops: Maximum relationship hops to explore (1=direct, 2=two-hop, etc.)
        per_table: If True, return suggestions organized per base table; if False, return combined results
        
    Returns:
        Dictionary containing join suggestions with connection paths
    """
    try:
        analyzer = _get_analyzer()
        
        # Parse base table names
        table_list = []
        for name in base_tables.split(','):
            name = name.strip()
            if name:
                table_list.append(name)
        
        if not table_list:
            return "Error: No base tables provided. Provide comma-separated table names like 'user_,booking'"
        
        logger.info(f"Getting join suggestions for tables: {table_list} (max_hops: {max_hops}, per_table: {per_table})")
        
        # Get suggestions - use combined mode for simplicity
        suggestions = analyzer.suggest_tables_for_join_combined(table_list, max_suggestions, max_hops)
        
        if not suggestions:
            return f"No join suggestions found within {max_hops} hops"
        
        # Return just comma-separated suggested table names
        analyzer.close()
        return ", ".join(suggestions)
        
    except Exception as e:
        logger.error(f"Error in suggest_joins: {e}")
        return f"Error: {str(e)}. Make sure Neo4j is running and the knowledge graph has been built"


@mcp.tool(description="Get detailed information about specific database views for statistics and reporting")
def explore_view(
    view_names: str = Field(description="Comma-separated list of view names to explore")
) -> str:
    """Get detailed information about specific database views for statistics and reporting.
    
    Args:
        view_names: Comma-separated list of view names to explore
        
    Returns:
        Dictionary containing view information including columns and dependencies
    """
    try:
        analyzer = _get_analyzer()
        
        # Parse view names (handle comma-separated input)
        view_list = []
        for name in view_names.split(','):
            name = name.strip()
            if name:
                view_list.append(name)
        
        if not view_list:
            return "Error: No view names provided"
        
        logger.info(f"Getting DDL for views: {view_list}")
        
        # Get view DDL from Neo4j - filter to views only
        tables_ddl = analyzer.backend.get_table_ddl(view_list)
        view_ddl = [table for table in tables_ddl if table.get('is_view', False) and not table.get('not_found', False)]
        missing_views = [table['name'] for table in tables_ddl if table.get('not_found', False)]
        non_views = [table['name'] for table in tables_ddl if not table.get('is_view', False) and not table.get('not_found', False)]
        
        # Format DDL output - one view per line, no newlines within DDL
        ddl_statements = []
        for view in view_ddl:
            clean_ddl = view['ddl'].replace('\n', ' ')
            ddl_statements.append(f"-- {view['name']} {clean_ddl}")
        
        result_text = "\n".join(ddl_statements)
        
        if missing_views:
            if result_text:
                result_text += f"\n\nWarning: Views not found in Neo4j: {', '.join(missing_views)}"
            else:
                result_text = f"Error: Views not found in Neo4j: {', '.join(missing_views)}\nMake sure Neo4j is running and the knowledge graph has been built with 'rkg analyze'"
        
        if non_views:
            if result_text:
                result_text += f"\n\nInfo: These are tables, not views: {', '.join(non_views)}. Use explore_table() instead."
        
        analyzer.close()
        return result_text
        
    except Exception as e:
        logger.error(f"Error in explore_view: {e}")
        return f"Error: {str(e)}\nMake sure Neo4j is running and the knowledge graph has been built with 'rkg analyze'"


@mcp.tool(description="Find database views related to specific tables for statistics and reporting queries")
def find_related_views(
    table_names: str = Field(description="Comma-separated list of table names to find related views for")
) -> str:
    """Find database views related to specific tables for statistics and reporting queries.
    
    Args:
        table_names: Comma-separated list of table names to find related views for
        
    Returns:
        Dictionary containing related views with their relationships to the tables
    """
    try:
        analyzer = _get_analyzer()
        
        # Parse table names
        table_list = []
        for name in table_names.split(','):
            name = name.strip()
            if name:
                table_list.append(name)
        
        if not table_list:
            return "Error: No table names provided. Provide comma-separated table names like 'user_,booking'"
        
        logger.info(f"Finding related views for tables: {table_list}")
        
        # Find views related to these tables using Neo4j
        related_views = set()
        with analyzer.backend.driver.session() as session:
            # Find views that depend on these tables (through view_dependency relationships)
            result = session.run("""
                UNWIND $table_names AS table_name
                MATCH (t {name: table_name})<-[:view_dependency]-(v)
                WHERE v.is_view = true
                RETURN DISTINCT v.name as view_name
            """, table_names=table_list)
            
            for record in result:
                related_views.add(record['view_name'])
            
            # Also find views connected through regular relationships
            result = session.run("""
                UNWIND $table_names AS table_name
                MATCH (t {name: table_name})-[]-(connected)
                WHERE connected.is_view = true
                RETURN DISTINCT connected.name as view_name
            """, table_names=table_list)
            
            for record in result:
                related_views.add(record['view_name'])
        
        # Get importance scores for ranking
        importance_scores = analyzer.backend.get_table_importance()
        
        # Sort by importance (return all views)
        sorted_views = sorted(
            related_views,
            key=lambda x: importance_scores.get(x, 0),
            reverse=True
        )
        
        if not sorted_views:
            return f"No views found related to tables: {', '.join(table_list)}"
        
        # Return comma-separated view names
        analyzer.close()
        return ", ".join(sorted_views)
        
    except Exception as e:
        logger.error(f"Error in find_related_views: {e}")
        return f"Error: {str(e)}. Make sure Neo4j is running and the knowledge graph has been built"


@mcp.tool(description="Execute SQL query safely with read-only validation and result truncation")
def execute_sql(
    sql: str = Field(description="SQL query to execute (read-only operations only)")
) -> str:
    """Execute SQL query safely with validation and result truncation.
    
    Args:
        sql: SQL query to execute (SELECT, EXPLAIN, SHOW, etc.)
        
    Returns:
        Formatted query results or error message
    """
    try:
        # Get database connection string from environment
        connection_string = os.getenv('DATABASE_URL')
        if not connection_string:
            return "Error: DATABASE_URL environment variable not set"
        
        # Get max tokens from environment (default 10000)
        max_tokens = int(os.getenv('SQL_MAX_TOKENS', '10000'))
        
        logger.info(f"Executing SQL query: {sql[:100]}..." if len(sql) > 100 else f"Executing SQL query: {sql}")
        
        # Create SQL executor and execute query
        executor = create_sql_executor(connection_string, max_tokens=max_tokens)
        try:
            result = executor.execute_query(sql)
            return result
        finally:
            executor.close()
            
    except Exception as e:
        logger.error(f"Error in execute_sql: {e}")
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()