"""REST API server for relational knowledge graph interactions."""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

from .analyzer import SchemaAnalyzer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Knowledge Graph API", description="REST API for database schema knowledge graph")

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


@app.get("/explore-table/{table_names}", response_class=PlainTextResponse)
def explore_table(table_names: str) -> str:
    """Get detailed DDL information about specific tables from Neo4j graph.
    
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
            raise HTTPException(status_code=400, detail="No table names provided")
        
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
        
        result_text = "\n".join(ddl_statements).strip()
        
        if missing_tables:
            if result_text:
                result_text += f"\n\nWarning: Tables not found in Neo4j: {', '.join(missing_tables)}"
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Tables not found in Neo4j: {', '.join(missing_tables)}. Make sure Neo4j is running and the knowledge graph has been built with 'rkg analyze'"
                )
        
        analyzer.close()
        return result_text
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in explore_table: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}. Make sure Neo4j is running and the knowledge graph has been built with 'rkg analyze'"
        )


@app.get("/clusters")
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
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}. Make sure Neo4j is running and clusters have been created with 'rkg create-clusters'"
        )


@app.get("/cluster/{cluster_id}", response_class=PlainTextResponse)
def show_cluster(cluster_id: str, detailed: bool = False, exclude_main: bool = True) -> str:
    """Show detailed information about a specific cluster.
    
    Args:
        cluster_id: The cluster ID to show details for
        detailed: Whether to include detailed column information for tables
        exclude_main: Whether to exclude tables that are in the main cluster
        
    Returns:
        Plain text cluster information and DDL if detailed=True
    """
    try:
        analyzer = _get_analyzer()
        
        if not cluster_id:
            raise HTTPException(status_code=400, detail="cluster_id is required")
        
        logger.info(f"Getting information for cluster: {cluster_id}")
        
        # Get cluster metadata and table list
        all_clusters = analyzer.backend.get_all_clusters()
        cluster_info = None
        for cluster in all_clusters:
            if cluster['id'] == cluster_id:
                cluster_info = cluster
                break
        
        if not cluster_info:
            raise HTTPException(
                status_code=404,
                detail=f"Cluster '{cluster_id}' not found. Use /clusters to see available cluster IDs"
            )
        
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
        
        # Build response text
        result_lines = []
        result_lines.append(f"Cluster: {cluster_id}")
        result_lines.append(f"Name: {cluster_info.get('name', 'N/A')}")
        result_lines.append(f"Description: {cluster_info.get('description', 'N/A')}")
        result_lines.append(f"Keywords: {', '.join(cluster_info.get('keywords', []))}")
        result_lines.append(f"Tables: {', '.join(cluster_tables)}")
        
        # Include DDL only if detailed=True
        if detailed:
            result_lines.append("\nDDL:")
            # Get cluster table DDL from Neo4j
            tables_ddl = analyzer.backend.get_cluster_tables_ddl(cluster_id)
            
            # Filter DDL results to match filtered table list
            filtered_ddl = [table for table in tables_ddl 
                          if table['name'] in cluster_tables]
            
            # Format DDL output
            for table in filtered_ddl:
                result_lines.append(f"\n-- {table['name']}")
                result_lines.append(table['ddl'])
        
        analyzer.close()
        return "\n".join(result_lines)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in show_cluster: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}. Make sure Neo4j is running and the cluster exists"
        )


@app.get("/main-cluster", response_class=PlainTextResponse)
def get_main_cluster(detailed: bool = False) -> str:
    """Get the main cluster (union of top N most important clusters).
    
    Args:
        detailed: Whether to include detailed DDL information
        
    Returns:
        Plain text main cluster information and tables
    """
    try:
        analyzer = _get_analyzer()
        
        # Get main cluster size from environment
        main_cluster_size = int(os.getenv('MAIN_CLUSTER_SIZE', '2'))
        
        logger.info(f"Getting main cluster from top {main_cluster_size} clusters")
        
        # Get all clusters ordered by importance (first clusters are most important)
        clusters = analyzer.backend.get_all_clusters()
        
        if not clusters:
            raise HTTPException(
                status_code=404,
                detail="No clusters found in the database. Use 'rkg create-clusters' to generate clusters first"
            )
        
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
        
        # Build response
        result_lines = []
        result_lines.append(f"Main Cluster (top {main_cluster_size} clusters combined)")
        result_lines.append(f"Total tables: {len(main_cluster_tables)}")
        result_lines.append(f"Tables: {', '.join(main_cluster_tables)}")
        
        if detailed:
            result_lines.append("\nDDL:")
            # Get DDL for all tables
            tables_ddl = analyzer.backend.get_table_ddl(main_cluster_tables)
            
            # Format DDL output
            for table in tables_ddl:
                if not table.get('not_found', False):
                    result_lines.append(f"\n-- {table['name']}")
                    result_lines.append(table['ddl'])
        
        analyzer.close()
        return "\n".join(result_lines)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_main_cluster: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}. Make sure Neo4j is running and clusters have been created"
        )


@app.get("/find-path/{table_names}")
def find_path(table_names: str, max_hops: int = 3) -> Dict[str, Any]:
    """Find all connection paths between the given tables.
    
    Args:
        table_names: Comma-separated list of tables to find connections between
        max_hops: Maximum relationship hops to explore (default: 3)
        
    Returns:
        Dictionary containing all connection paths
    """
    try:
        analyzer = _get_analyzer()
        
        # Parse table names
        table_list = []
        for name in table_names.split(','):
            name = name.strip()
            if name:
                table_list.append(name)
        
        if len(table_list) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 tables are required to find connections. Provide comma-separated table names like 'user_,sample,booking'"
            )
        
        logger.info(f"Finding connections between tables: {table_list} (max_hops: {max_hops})")
        
        # Find all connections
        connections = analyzer.backend.find_all_connections(table_list, max_hops)
        
        if not connections:
            return {
                "success": True,
                "connections": [],
                "message": f"No connections found between these tables within {max_hops} hops"
            }
        
        # Format connections with readable paths
        formatted_connections = []
        for conn in connections:
            path_str = " - ".join(conn['path'])
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in find_path: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}. Make sure Neo4j is running and the knowledge graph has been built"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)