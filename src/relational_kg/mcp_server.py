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
        
        # Get cluster metadata and table list
        all_clusters = analyzer.backend.get_all_clusters()
        cluster_info = None
        for cluster in all_clusters:
            if cluster['id'] == cluster_id:
                cluster_info = cluster
                break
        
        if not cluster_info:
            return {"result": {
                "success": False,
                "error": f"Cluster '{cluster_id}' not found",
                "help": "Use list_clusters to see available cluster IDs"
            }}
        
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
        
        result = {
            "success": True,
            "cluster_id": cluster_id,
            "tables": cluster_tables
        }
        
        # Add cluster metadata
        result["cluster_name"] = cluster_info.get('name', '')
        result["cluster_description"] = cluster_info.get('description', '')
        result["cluster_keywords"] = cluster_info.get('keywords', [])
        
        # Include DDL only if detailed=True
        if detailed:
            # Get cluster table DDL from Neo4j
            tables_ddl = analyzer.backend.get_cluster_tables_ddl(cluster_id)
            
            # Filter DDL results to match filtered table list
            filtered_ddl = [table for table in tables_ddl 
                          if table['name'] in cluster_tables]
            
            # Format DDL output
            ddl_statements = []
            for table in filtered_ddl:
                ddl_statements.append(f"-- {table['name']}")
                ddl_statements.append(table['ddl'])
                ddl_statements.append("")  # Empty line between tables
            
            result["ddl"] = "\n".join(ddl_statements).strip()
        
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


@mcp.tool()
def find_path(tables: str, max_hops: int = 3) -> Dict[str, Any]:
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
            return {
                "success": False,
                "error": "At least 2 tables are required to find connections",
                "help": "Provide comma-separated table names like 'user_,sample,booking'"
            }
        
        logger.info(f"Finding connections between tables: {table_list} (max_hops: {max_hops})")
        
        # Find all connections
        connections = analyzer.backend.find_all_connections(table_list, max_hops)
        
        if not connections:
            return {
                "success": True,
                "connections": [],
                "message": f"No connections found between these tables within {max_hops} hops"
            }
        
        # Format connections with readable paths (simplified)
        formatted_connections = []
        for conn in connections:
            path_str = " → ".join(conn['path'])
            formatted_connections.append({
                "table1": conn['table1'],
                "table2": conn['table2'],
                "path": conn['path'],
                "path_string": path_str
            })
        
        result = {
            "success": True,
            "connections": formatted_connections
        }
        
        analyzer.close()
        return result
        
    except Exception as e:
        logger.error(f"Error in find_path: {e}")
        return {
            "success": False,
            "error": str(e),
            "help": "Make sure Neo4j is running and the knowledge graph has been built"
        }


@mcp.tool()
def suggest_joins(base_tables: str, max_suggestions: int = 5, max_hops: int = 1, per_table: bool = False) -> Dict[str, Any]:
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
            return {
                "success": False,
                "error": "No base tables provided",
                "help": "Provide comma-separated table names like 'user_,booking'"
            }
        
        logger.info(f"Getting join suggestions for tables: {table_list} (max_hops: {max_hops}, per_table: {per_table})")
        
        if per_table:
            # Per-table suggestions
            suggestions_per_table = analyzer.suggest_tables_for_join(table_list, max_suggestions, max_hops)
            
            # Format with connection paths
            formatted_suggestions = {}
            for base_table, suggestions in suggestions_per_table.items():
                formatted_suggestions[base_table] = []
                for suggested_table in suggestions:
                    # Find connection path
                    path = analyzer.find_connection_path(base_table, suggested_table, max_hops)
                    if path:
                        distance = len(path) - 1
                        distance_desc = "direct" if distance == 1 else f"{distance}-hop"
                        path_str = " → ".join(path)
                    else:
                        distance = None
                        distance_desc = "unknown"
                        path_str = f"{base_table} ↔ {suggested_table}"
                    
                    formatted_suggestions[base_table].append({
                        "table": suggested_table,
                        "path": path,
                        "path_string": path_str,
                        "distance": distance,
                        "distance_description": distance_desc
                    })
            
            result = {
                "success": True,
                "mode": "per_table",
                "base_tables": table_list,
                "max_suggestions": max_suggestions,
                "max_hops": max_hops,
                "suggestions_per_table": formatted_suggestions,
                "total_unique_suggestions": len(set(
                    suggestion["table"] 
                    for suggestions in formatted_suggestions.values() 
                    for suggestion in suggestions
                ))
            }
        else:
            # Combined suggestions
            suggestions = analyzer.suggest_tables_for_join_combined(table_list, max_suggestions, max_hops)
            
            if not suggestions:
                return {
                    "success": True,
                    "mode": "combined",
                    "base_tables": table_list,
                    "suggestions": [],
                    "message": f"No join suggestions found within {max_hops} hops"
                }
            
            # Format with connection paths from each base table
            formatted_suggestions = []
            for suggested_table in suggestions:
                suggestion_entry = {
                    "table": suggested_table,
                    "paths": []
                }
                
                # Find paths from each base table
                for base_table in table_list:
                    path = analyzer.find_connection_path(base_table, suggested_table, max_hops)
                    if path and len(path) > 1:
                        distance = len(path) - 1
                        distance_desc = "direct" if distance == 1 else f"{distance}-hop"
                        path_str = " → ".join(path)
                        suggestion_entry["paths"].append({
                            "from_table": base_table,
                            "path": path,
                            "path_string": path_str,
                            "distance": distance,
                            "distance_description": distance_desc
                        })
                
                formatted_suggestions.append(suggestion_entry)
            
            result = {
                "success": True,
                "mode": "combined",
                "base_tables": table_list,
                "max_suggestions": max_suggestions,
                "max_hops": max_hops,
                "suggestions": formatted_suggestions,
                "total_suggestions": len(formatted_suggestions)
            }
        
        analyzer.close()
        return result
        
    except Exception as e:
        logger.error(f"Error in suggest_joins: {e}")
        return {
            "success": False,
            "error": str(e),
            "help": "Make sure Neo4j is running and the knowledge graph has been built"
        }


@mcp.tool()
def explore_view(view_names: str, detailed: bool = True) -> Dict[str, Any]:
    """Get detailed information about specific database views for statistics and reporting.
    
    Args:
        view_names: Comma-separated list of view names to explore
        detailed: Whether to include detailed column information
        
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
            return {"error": "No view names provided"}
        
        logger.info(f"Getting DDL for views: {view_list}")
        
        # Get view DDL from Neo4j - filter to views only
        tables_ddl = analyzer.backend.get_table_ddl(view_list)
        view_ddl = [table for table in tables_ddl if table.get('is_view', False) and not table.get('not_found', False)]
        missing_views = [table['name'] for table in tables_ddl if table.get('not_found', False)]
        non_views = [table['name'] for table in tables_ddl if not table.get('is_view', False) and not table.get('not_found', False)]
        
        # Format DDL output
        ddl_statements = []
        for view in view_ddl:
            ddl_statements.append(f"-- {view['name']} (VIEW)")
            ddl_statements.append(view['ddl'])
            ddl_statements.append("")  # Empty line between views
        
        result = {
            "success": True,
            "views_requested": len(view_list),
            "views_found": len(view_ddl),
            "ddl": "\n".join(ddl_statements).strip()
        }
        
        if missing_views:
            result["missing_views"] = missing_views
            result["warning"] = f"Views not found in Neo4j: {', '.join(missing_views)}"
        
        if non_views:
            result["non_views"] = non_views
            result["info"] = f"These are tables, not views: {', '.join(non_views)}. Use explore_table() instead."
        
        analyzer.close()
        return {"result": result}
        
    except Exception as e:
        logger.error(f"Error in explore_view: {e}")
        return {"result": {
            "success": False,
            "error": str(e),
            "help": "Make sure Neo4j is running and the knowledge graph has been built with 'rkg analyze'"
        }}


@mcp.tool()
def find_related_views(table_names: str, max_suggestions: int = 5) -> Dict[str, Any]:
    """Find database views related to specific tables for statistics and reporting queries.
    
    Args:
        table_names: Comma-separated list of table names to find related views for
        max_suggestions: Maximum number of view suggestions to return
        
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
            return {
                "success": False,
                "error": "No table names provided",
                "help": "Provide comma-separated table names like 'user_,booking'"
            }
        
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
        
        # Sort by importance and limit results
        sorted_views = sorted(
            related_views,
            key=lambda x: importance_scores.get(x, 0),
            reverse=True
        )[:max_suggestions]
        
        if not sorted_views:
            return {
                "success": True,
                "related_views": [],
                "message": f"No views found related to tables: {', '.join(table_list)}"
            }
        
        result = {
            "success": True,
            "base_tables": table_list,
            "related_views": sorted_views,
            "total_found": len(sorted_views)
        }
        
        analyzer.close()
        return result
        
    except Exception as e:
        logger.error(f"Error in find_related_views: {e}")
        return {
            "success": False,
            "error": str(e),
            "help": "Make sure Neo4j is running and the knowledge graph has been built"
        }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()