# Model Context Protocol (MCP) Configuration

This document explains how to configure and use the Knowledge Graph MCP server with Claude Desktop and other MCP-compatible clients.

## Prerequisites

1. **Neo4j Database**: Make sure Neo4j is running and accessible
2. **Knowledge Graph Built**: Run `rkg analyze` to build the knowledge graph in Neo4j
3. **Clusters Created** (optional): Run `rkg create-clusters` for cluster-related tools

## Environment Variables

Set these environment variables for Neo4j connection:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j" 
export NEO4J_PASSWORD="your_password"
```

## Claude Desktop Configuration

Add this to your Claude Desktop configuration file (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "uvx",
      "args": [
        "--from", "/path/to/knowledge-graph-schema",
        "python", "-m", "relational_kg.mcp_server"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your_neo4j_password"
      }
    }
  }
}
```

**Alternative using direct Python execution:**
```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "python",
      "args": ["-m", "relational_kg.mcp_server"],
      "cwd": "/path/to/knowledge-graph-schema",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j", 
        "NEO4J_PASSWORD": "your_neo4j_password"
      }
    }
  }
}
```

## Development and Testing

For development, you can use the MCP Inspector:

```bash
# Install the project in development mode
pip install -e .

# Test the MCP server with inspector
mcp dev src/relational_kg/mcp_server.py
```

## Available Tools

### 1. explore_table
Get detailed information about specific tables from the Neo4j knowledge graph.

**Parameters:**
- `table_names` (string): Comma-separated list of table names
- `detailed` (boolean, default: true): Include detailed column information

**Example usage in Claude:**
"Use the explore_table tool to get details about the 'users' and 'orders' tables"

### 2. list_clusters  
List all available table clusters.

**Parameters:** None

**Example usage in Claude:**
"Use the list_clusters tool to show me all available table clusters"

### 3. show_cluster
Show detailed information about a specific cluster.

**Parameters:**
- `cluster_id` (string): The cluster ID to show details for
- `detailed` (boolean, default: false): Include detailed column information

**Example usage in Claude:**
"Use the show_cluster tool to show me details for cluster 'user_management'"

## Troubleshooting

### Common Issues

1. **"NEO4J_PASSWORD environment variable is required"**
   - Make sure the NEO4J_PASSWORD is set in your configuration

2. **"Tables not found in Neo4j"**
   - Run `rkg analyze` to build the knowledge graph first
   - Verify Neo4j connection settings

3. **"No clusters found"**
   - Run `rkg create-clusters` to generate table clusters first

4. **MCP Connection Issues**
   - Check that the file paths in configuration are correct
   - Ensure all dependencies are installed: `pip install -e .`
   - Verify Neo4j is running and accessible

### Debug Logging

The MCP server includes detailed logging. Check the logs for connection and operation details.

## Security Notes

- The MCP server only connects to Neo4j (read-only operations)
- No direct database access is required
- Environment variables keep credentials secure
- All operations are performed through the existing Neo4j backend