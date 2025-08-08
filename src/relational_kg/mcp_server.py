"""Model Context Protocol (MCP) server for relational knowledge graph interactions."""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .analyzer import SchemaAnalyzer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize MCP server
if not MCP_AVAILABLE:
    logger.error("MCP package not available. Install with: pip install 'mcp[cli]'")
    sys.exit(1)

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


@mcp.tool()
def explore_table(table_names: str, detailed: bool = True) -> Dict[str, Any]:
    """Get detailed information about specific tables from Neo4j graph.
    
    Args:
        table_names: Comma-separated list of table names to explore
        detailed: Whether to include detailed column information
        
    Returns:
        Dictionary containing table information including columns and foreign keys
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
            return {"error": "No table names provided"}
        
        logger.info(f"Getting DDL for tables: {table_list}")
        
        # Get table DDL from Neo4j
        tables_ddl = analyzer.backend.get_table_ddl(table_list)
        
        # Separate found and missing tables  
        found_tables = [t for t in tables_ddl if not t.get('not_found', False)]
        missing_tables = [t['name'] for t in tables_ddl if t.get('not_found', False)]
        
        # Format DDL output
        ddl_statements = []
        for table in found_tables:
            ddl_statements.append(f"-- {table['name']}")
            ddl_statements.append(table['ddl'])
            ddl_statements.append("")  # Empty line between tables
        
        result = {
            "success": True,
            "tables_requested": len(table_list),
            "tables_found": len(found_tables),
            "ddl": "\n".join(ddl_statements).strip()
        }
        
        if missing_tables:
            result["missing_tables"] = missing_tables
            result["warning"] = f"Tables not found in Neo4j: {', '.join(missing_tables)}"
        
        analyzer.close()
        return {"result": result}
        
    except Exception as e:
        logger.error(f"Error in explore_table: {e}")
        return {"result": {
            "success": False,
            "error": str(e),
            "help": "Make sure Neo4j is running and the knowledge graph has been built with 'rkg analyze'"
        }}


@mcp.tool() 
def list_clusters(exclude_main: bool = True) -> Dict[str, Any]:
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
                # Skip the first N clusters (they make up the main cluster)
                clusters = clusters[main_cluster_size:]
        
        result = {
            "success": True,
            "cluster_count": len(clusters),
            "clusters": clusters
        }
        
        if not clusters:
            result["info"] = "No clusters found. Run 'rkg create-clusters' to generate clusters first."
        
        analyzer.close()
        return result
        
    except Exception as e:
        logger.error(f"Error in list_clusters: {e}")
        return {
            "success": False,
            "error": str(e),
            "help": "Make sure Neo4j is running and clusters have been created with 'rkg create-clusters'"
        }


@mcp.tool()
def show_cluster(cluster_id: str, detailed: bool = False, exclude_main: bool = True) -> Dict[str, Any]:
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
        
        # Get cluster table DDL from Neo4j
        tables_ddl = analyzer.backend.get_cluster_tables_ddl(cluster_id)
        
        if not tables_ddl:
            return {"result": {
                "success": False,
                "error": f"Cluster '{cluster_id}' not found",
                "help": "Use list_clusters to see available cluster IDs"
            }}
        
        # Filter out main cluster tables if exclude_main is True
        if exclude_main:
            # Get main cluster size from environment
            main_cluster_size = int(os.getenv('MAIN_CLUSTER_SIZE', '2'))
            
            # Get all clusters to find main cluster tables
            all_clusters = analyzer.backend.get_all_clusters()
            main_cluster_tables = set()
            
            if all_clusters and len(all_clusters) >= main_cluster_size:
                for i in range(main_cluster_size):
                    cluster_tables = all_clusters[i].get('tables', [])
                    main_cluster_tables.update(cluster_tables)
            
            # Filter out main cluster tables
            tables_ddl = [table for table in tables_ddl 
                         if table['name'] not in main_cluster_tables]
        
        # Get cluster metadata
        all_clusters = analyzer.backend.get_all_clusters()
        cluster_info = None
        for cluster in all_clusters:
            if cluster['id'] == cluster_id:
                cluster_info = cluster
                break
        
        # Format DDL output
        ddl_statements = []
        for table in tables_ddl:
            ddl_statements.append(f"-- {table['name']}")
            ddl_statements.append(table['ddl'])
            ddl_statements.append("")  # Empty line between tables
        
        result = {
            "success": True,
            "cluster_id": cluster_id,
            "table_count": len(tables_ddl),
            "ddl": "\n".join(ddl_statements).strip()
        }
        
        # Add cluster metadata if found
        if cluster_info:
            result["cluster_name"] = cluster_info.get('name', '')
            result["cluster_description"] = cluster_info.get('description', '')
            result["cluster_keywords"] = cluster_info.get('keywords', [])
        
        analyzer.close()
        return {"result": result}
        
    except Exception as e:
        logger.error(f"Error in show_cluster: {e}")
        return {"result": {
            "success": False,
            "error": str(e),
            "help": "Make sure Neo4j is running and the cluster exists"
        }}


@mcp.tool()
def get_main_cluster(detailed: bool = False) -> Dict[str, Any]:
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
            return {
                "success": False,
                "error": "No clusters found in the database",
                "help": "Use 'rkg create-clusters' to generate clusters first"
            }
        
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
            
            # Format DDL output
            ddl_statements = []
            for table in tables_ddl:
                if not table.get('not_found', False):
                    ddl_statements.append(f"-- {table['name']}")
                    ddl_statements.append(table['ddl'])
                    ddl_statements.append("")  # Empty line between tables
            
            result_ddl = "\n".join(ddl_statements).strip()
        else:
            result_ddl = None
        
        result = {
            "success": True,
            "main_cluster_size": main_cluster_size,
            "used_clusters": used_clusters,
            "table_count": len(main_cluster_tables),
            "tables": main_cluster_tables
        }
        
        if detailed:
            result["ddl"] = result_ddl
        
        analyzer.close()
        return result
        
    except Exception as e:
        logger.error(f"Error in get_main_cluster: {e}")
        return {
            "success": False,
            "error": str(e),
            "help": "Make sure Neo4j is running and clusters have been created"
        }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()