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
        
        logger.info(f"Getting details for tables: {table_list}")
        
        # Get table details from Neo4j
        tables = analyzer.backend.get_table_details(table_list, detailed=detailed)
        
        # Separate found and missing tables  
        found_tables = [t for t in tables if not t.get('not_found', False)]
        missing_tables = [t['name'] for t in tables if t.get('not_found', False)]
        
        result = {
            "success": True,
            "tables_requested": len(table_list),
            "tables_found": len(found_tables),
            "tables": found_tables,
            "result": found_tables
        }
        
        if missing_tables:
            result["missing_tables"] = missing_tables
            result["warning"] = f"Tables not found in Neo4j: {', '.join(missing_tables)}"
        
        analyzer.close()
        return result
        
    except Exception as e:
        logger.error(f"Error in explore_table: {e}")
        return {
            "success": False,
            "error": str(e),
            "help": "Make sure Neo4j is running and the knowledge graph has been built with 'rkg analyze'"
        }


@mcp.tool() 
def list_clusters() -> Dict[str, Any]:
    """List all available table clusters from Neo4j.
    
    Returns:
        Dictionary containing all clusters with their basic information
    """
    try:
        analyzer = _get_analyzer()
        
        logger.info("Retrieving all clusters from Neo4j")
        
        # Get all clusters from Neo4j
        clusters = analyzer.backend.get_all_clusters()
        
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
def show_cluster(cluster_id: str, detailed: bool = False) -> Dict[str, Any]:
    """Show detailed information about a specific cluster.
    
    Args:
        cluster_id: The cluster ID to show details for
        detailed: Whether to include detailed column information for tables
        
    Returns:
        Dictionary containing detailed cluster information and its tables
    """
    try:
        analyzer = _get_analyzer()
        
        if not cluster_id:
            return {"error": "cluster_id is required"}
        
        logger.info(f"Getting details for cluster: {cluster_id}")
        
        # Get cluster tables from Neo4j
        tables = analyzer.backend.get_cluster_tables(cluster_id, detailed=detailed)
        
        if not tables:
            return {
                "success": False,
                "error": f"Cluster '{cluster_id}' not found",
                "help": "Use list_clusters to see available cluster IDs"
            }
        
        # Get cluster metadata
        all_clusters = analyzer.backend.get_all_clusters()
        cluster_info = None
        for cluster in all_clusters:
            if cluster['id'] == cluster_id:
                cluster_info = cluster
                break
        
        result = {
            "success": True,
            "cluster_id": cluster_id,
            "table_count": len(tables),
            "tables": tables,
            "result": tables
        }
        
        # Add cluster metadata if found
        if cluster_info:
            result["cluster_name"] = cluster_info.get('name', '')
            result["cluster_description"] = cluster_info.get('description', '')
            result["cluster_keywords"] = cluster_info.get('keywords', [])
        
        analyzer.close()
        return result
        
    except Exception as e:
        logger.error(f"Error in show_cluster: {e}")
        return {
            "success": False,
            "error": str(e),
            "help": "Make sure Neo4j is running and the cluster exists"
        }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()