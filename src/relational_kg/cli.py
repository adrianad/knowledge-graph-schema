"""Command-line interface for the relational knowledge graph."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from .analyzer import SchemaAnalyzer
from .visualizer import GraphVisualizer
from .llm_extractor import LLMKeywordExtractor
from .llm_cluster_analyzer import LLMClusterAnalyzer


# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def _create_analyzer(connection: str, backend: str = 'networkx', 
                    neo4j_uri: str = None, neo4j_user: str = None, 
                    neo4j_password: str = None) -> SchemaAnalyzer:
    """Create analyzer with specified backend."""
    backend_kwargs = {}
    if backend == 'neo4j':
        # Use environment variables as defaults
        uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        
        if not password:
            click.echo("Neo4j backend requires NEO4J_PASSWORD environment variable or --neo4j-password option")
            sys.exit(1)
            
        backend_kwargs = {
            'uri': uri,
            'username': user,
            'password': password
        }
    
    return SchemaAnalyzer(connection, backend=backend, **backend_kwargs)


def backend_options(f):
    """Decorator to add backend options to commands."""
    f = click.option('--neo4j-password', help='Neo4j password (overrides NEO4J_PASSWORD env var)')(f)
    f = click.option('--neo4j-user', help='Neo4j username (overrides NEO4J_USER env var)')(f)
    f = click.option('--neo4j-uri', help='Neo4j connection URI (overrides NEO4J_URI env var)')(f)
    f = click.option('--backend', '-b', default='neo4j', help='Graph backend (networkx, neo4j)')(f)
    f = click.option('--include-views/--exclude-views', default=True, help='Include database views in analysis')(f)
    return f


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose: bool) -> None:
    """Relational Knowledge Graph CLI for database schema analysis."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.option('--connection', '-c', help='Database connection string (overrides DATABASE_URL env var)')
@click.option('--output', '-o', help='Output file for graph data (JSON)')
@backend_options
def analyze(connection: str, output: Optional[str], backend: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, include_views: bool) -> None:
    """Analyze database schema and build knowledge graph."""
    try:
        # Use DATABASE_URL from environment if connection not provided
        connection_final = connection or os.getenv('DATABASE_URL')
        if not connection_final:
            click.echo("❌ Database connection required: use -c/--connection or set DATABASE_URL environment variable", err=True)
            sys.exit(1)
            
        analyzer = _create_analyzer(connection_final, backend, neo4j_uri, neo4j_user, neo4j_password)
        click.echo(f"Analyzing database schema using {backend} backend...")
        
        analyzer.analyze_schema(include_views=include_views)
        
        click.echo(f"✅ Analysis complete!")
        click.echo(f"📊 Total tables: {len(analyzer.tables)}")
        
        # Get basic stats only (avoid expensive operations)
        try:
            stats = analyzer.backend.get_graph_stats()
            click.echo(f"🔗 Graph edges: {stats['edge_count']}")
            click.echo(f"🔗 Graph nodes: {stats['node_count']}")
        except Exception as e:
            click.echo(f"⚠️  Could not get graph stats: {e}")
        
        # Skip expensive summary operations for large schemas
        if len(analyzer.tables) < 100:
            try:
                summary = analyzer.get_schema_summary()
                click.echo(f"🏘️  Table clusters: {len(summary['table_clusters'])}")
                
                # Show most important tables
                click.echo("\\n🌟 Most important tables:")
                for table_info in summary['most_important_tables'][:5]:
                    click.echo(f"  • {table_info['table']} (score: {table_info['importance_score']:.3f})")
            except Exception as e:
                click.echo(f"⚠️  Skipped detailed analysis for large schema: {e}")
        else:
            click.echo(f"⚠️  Skipped detailed analysis for large schema ({len(analyzer.tables)} tables)")
        
        # Save graph data if requested
        if output:
            if hasattr(analyzer.backend, 'export_graph_data'):
                graph_data = analyzer.backend.export_graph_data()
                with open(output, 'w') as f:
                    json.dump(graph_data, f, indent=2)
                click.echo(f"💾 Graph data saved to {output}")
            else:
                click.echo("⚠️  Graph data export not supported for this backend")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', help='Database connection string (overrides DATABASE_URL env var)')
@click.option('--keywords', '-k', required=True, help='Comma-separated keywords to search for')
@click.option('--max-tables', '-m', default=10, help='Maximum number of tables to return')
@click.option('--include-related', '-r', is_flag=True, help='Include related tables in results')
@backend_options
def find_tables(connection: str, keywords: str, max_tables: int, include_related: bool, backend: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, include_views: bool) -> None:
    """Find relevant tables based on keywords."""
    try:
        # Use DATABASE_URL from environment if connection not provided
        connection_final = connection or os.getenv('DATABASE_URL')
        if not connection_final:
            click.echo("❌ Database connection required: use -c/--connection or set DATABASE_URL environment variable", err=True)
            sys.exit(1)
            
        analyzer = _create_analyzer(connection_final, backend, neo4j_uri, neo4j_user, neo4j_password)
        analyzer.analyze_schema(include_views=include_views)
        
        keyword_list = [k.strip() for k in keywords.split(',')]
        click.echo(f"🔍 Searching for tables matching: {', '.join(keyword_list)}")
        
        relevant_tables = analyzer.find_relevant_tables(keyword_list, max_tables)
        
        if not relevant_tables:
            click.echo("❌ No relevant tables found")
            return
        
        click.echo(f"\\n📋 Found {len(relevant_tables)} relevant tables:")
        
        all_tables = set()
        
        for table_score in relevant_tables:
            click.echo(f"\\n• {table_score.table_name} (score: {table_score.score:.1f})")
            for reason in table_score.reasons:
                click.echo(f"  └─ {reason}")
            
            all_tables.add(table_score.table_name)
            
            if include_related:
                related = analyzer.get_table_cluster(table_score.table_name)
                related.discard(table_score.table_name)  # Remove the main table
                if related:
                    click.echo(f"  🔗 Related tables: {', '.join(sorted(related))}")
                    all_tables.update(related)
        
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', help='Database connection string (overrides DATABASE_URL env var)')
@click.option('--tables', '-t', required=True, help='Comma-separated list of base tables')
@click.option('--max-suggestions', '-m', default=5, help='Maximum number of suggestions')
@backend_options
def suggest_joins(connection: str, tables: str, max_suggestions: int, backend: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, include_views: bool) -> None:
    """Suggest tables that could be joined with the given base tables."""
    try:
        # Use DATABASE_URL from environment if connection not provided
        connection_final = connection or os.getenv('DATABASE_URL')
        if not connection_final:
            click.echo("❌ Database connection required: use -c/--connection or set DATABASE_URL environment variable", err=True)
            sys.exit(1)
            
        analyzer = _create_analyzer(connection_final, backend, neo4j_uri, neo4j_user, neo4j_password)
        analyzer.analyze_schema(include_views=include_views)
        
        base_tables = [t.strip() for t in tables.split(',')]
        click.echo(f"🔗 Finding join suggestions for: {', '.join(base_tables)}")
        
        suggestions = analyzer.suggest_tables_for_join(base_tables, max_suggestions)
        
        if not suggestions:
            click.echo("❌ No join suggestions found")
            return
        
        click.echo(f"\\n💡 Suggested tables to join:")
        for i, table in enumerate(suggestions, 1):
            click.echo(f"{i}. {table}")
            
            # Show connection paths
            for base_table in base_tables:
                path = analyzer.find_connection_path(base_table, table)
                if path and len(path) > 1:
                    path_str = " → ".join(path)
                    click.echo(f"   └─ Path from {base_table}: {path_str}")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', help='Database connection string (overrides DATABASE_URL env var)')
@click.option('--output', '-o', default='schema_graph.html', help='Output HTML file')
@click.option('--layout', '-l', default='spring', help='Graph layout (spring, circular, hierarchical)')
@backend_options
def visualize(connection: str, output: str, layout: str, backend: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, include_views: bool) -> None:
    """Generate interactive visualization of the schema graph."""
    try:
        # Use DATABASE_URL from environment if connection not provided
        connection_final = connection or os.getenv('DATABASE_URL')
        if not connection_final:
            click.echo("❌ Database connection required: use -c/--connection or set DATABASE_URL environment variable", err=True)
            sys.exit(1)
            
        analyzer = _create_analyzer(connection_final, backend, neo4j_uri, neo4j_user, neo4j_password)
        analyzer.analyze_schema(include_views=include_views)
        
        click.echo("🎨 Generating visualization...")
        
        if backend == 'networkx' and hasattr(analyzer.backend, 'schema_graph'):
            visualizer = GraphVisualizer(analyzer.backend.schema_graph)
            visualizer.create_interactive_plot(output, layout_type=layout)
        else:
            click.echo("⚠️  Visualization currently only supports NetworkX backend")
            return
        
        click.echo(f"✅ Visualization saved to {output}")
        click.echo(f"🌐 Open {output} in your browser to view the interactive graph")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', help='Database connection string (overrides DATABASE_URL env var)')
@backend_options
def summary(connection: str, backend: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, include_views: bool) -> None:
    """Display summary statistics of the database schema."""
    try:
        # Use DATABASE_URL from environment if connection not provided
        connection_final = connection or os.getenv('DATABASE_URL')
        if not connection_final:
            click.echo("❌ Database connection required: use -c/--connection or set DATABASE_URL environment variable", err=True)
            sys.exit(1)
            
        analyzer = _create_analyzer(connection_final, backend, neo4j_uri, neo4j_user, neo4j_password)
        analyzer.analyze_schema(include_views=include_views)
        
        summary = analyzer.get_schema_summary()
        
        click.echo("📊 Database Schema Summary")
        click.echo("=" * 40)
        click.echo(f"Database Type: {summary['database_type']}")
        
        # Show separate counts if available
        if 'total_entities' in summary:
            click.echo(f"Total Tables: {summary['total_tables']}")
            click.echo(f"Total Views: {summary['total_views']}")
            click.echo(f"Total Entities: {summary['total_entities']}")
        else:
            click.echo(f"Total Tables: {summary['total_tables']}")
        
        
        stats = summary['graph_statistics']
        click.echo(f"Graph Edges: {stats['edge_count']}")
        click.echo(f"Graph Density: {stats['density']:.3f}")
        click.echo(f"Is Connected: {stats['is_connected']}")
        if 'strongly_connected_components' in stats:
            click.echo(f"Connected Components: {stats['strongly_connected_components']}")
        
        click.echo(f"\\n🏘️  Table Clusters ({len(summary['table_clusters'])}):")
        for i, cluster in enumerate(summary['table_clusters'], 1):
            if len(cluster) > 1:  # Only show clusters with multiple tables
                click.echo(f"  {i}. {', '.join(sorted(cluster))}")
        
        # Show separate importance lists if available
        if 'most_important_views' in summary:
            click.echo("\\n🗃️  Most Important Tables:")
            for table_info in summary['most_important_tables'][:20]:
                click.echo(f"  • {table_info['table']} (score: {table_info['importance_score']:.3f})")
            
            if summary['most_important_views']:
                click.echo("\\n👁️  Most Important Views:")
                for view_info in summary['most_important_views'][:10]:
                    click.echo(f"  • {view_info['table']} (score: {view_info['importance_score']:.3f})")
            else:
                click.echo("\\n👁️  No views found or views have no connections")
        else:
            click.echo("\\n🌟 Most Important Tables:")
            for table_info in summary['most_important_tables'][:20]:
                click.echo(f"  • {table_info['table']} (score: {table_info['importance_score']:.3f})")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', help='Database connection string (overrides DATABASE_URL env var)')
@click.option('--include-views/--exclude-views', default=True, help='Include database views in keyword extraction')
@click.option('--max-concurrent', type=int, help='Maximum concurrent LLM requests (overrides LLM_MAX_CONCURRENT env var)')
@click.option('--neo4j-uri', help='Neo4j connection URI (overrides NEO4J_URI env var)')
@click.option('--neo4j-user', help='Neo4j username (overrides NEO4J_USER env var)')
@click.option('--neo4j-password', help='Neo4j password (overrides NEO4J_PASSWORD env var)')
def llm_keyword_extraction(connection: str, include_views: bool, max_concurrent: int, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
    """Extract business keywords from tables/views using LLM and store in Neo4j."""
    try:
        # Use DATABASE_URL from environment if connection not provided
        connection_final = connection or os.getenv('DATABASE_URL')
        if not connection_final:
            click.echo("❌ Database connection required: use -c/--connection or set DATABASE_URL environment variable", err=True)
            sys.exit(1)
            
        # Initialize analyzer with Neo4j backend (required for keyword storage)
        analyzer = _create_analyzer(connection_final, 'neo4j', neo4j_uri, neo4j_user, neo4j_password)
        
        click.echo("🔍 Analyzing database schema...")
        analyzer.analyze_schema(include_views=include_views)
        
        click.echo("🤖 Initializing LLM keyword extractor...")
        extractor = LLMKeywordExtractor()
        
        # Get max_concurrent from environment if not provided
        max_concurrent_final = max_concurrent or int(os.getenv('LLM_MAX_CONCURRENT', '10'))
        
        # Extract keywords for all tables/views using async processing
        click.echo(f"📝 Extracting keywords from tables and views (max {max_concurrent_final} concurrent requests)...")
        keyword_results = extractor.extract_keywords_batch_sync(analyzer.tables, include_views=include_views, max_concurrent=max_concurrent_final)
        
        # Store keywords in Neo4j
        click.echo("💾 Storing keywords in Neo4j...")
        stored_count = 0
        for result in keyword_results:
            if result.keywords or result.business_concepts:
                analyzer.backend.add_keywords_to_table(
                    result.table_name,
                    result.keywords,
                    result.business_concepts
                )
                stored_count += 1
                
                # Show progress
                if stored_count % 10 == 0:
                    click.echo(f"  Stored keywords for {stored_count} entities...")
        
        click.echo(f"✅ Keyword extraction complete!")
        click.echo(f"📊 Processed {len(keyword_results)} entities")
        click.echo(f"💾 Stored keywords for {stored_count} entities")
        
        # Show some statistics
        total_keywords = sum(len(r.keywords) for r in keyword_results)
        total_concepts = sum(len(r.business_concepts) for r in keyword_results)
        click.echo(f"🏷️  Total keywords extracted: {total_keywords}")
        click.echo(f"🏢 Total business concepts: {total_concepts}")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', help='Database connection string (overrides DATABASE_URL env var)')
@click.option('--table', '-t', required=True, help='Table name to explore relationships from')
@click.option('--hops', '-h', default=2, help='Number of relationship hops to explore (default: 2)')
@click.option('--show-views/--hide-views', default=False, help='Show views in the relationship exploration results')
@backend_options
def explore_table(connection: str, table: str, hops: int, show_views: bool, backend: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, include_views: bool) -> None:
    """Explore all relationships from a specific table within N hops."""
    try:
        # Use DATABASE_URL from environment if connection not provided
        connection_final = connection or os.getenv('DATABASE_URL')
        if not connection_final:
            click.echo("❌ Database connection required: use -c/--connection or set DATABASE_URL environment variable", err=True)
            sys.exit(1)
            
        analyzer = _create_analyzer(connection_final, backend, neo4j_uri, neo4j_user, neo4j_password)
        analyzer.analyze_schema(include_views=include_views)
        
        click.echo(f"🔍 Exploring relationships from table '{table}' within {hops} hops...")
        
        # Check if table exists
        if table not in analyzer.tables:
            click.echo(f"❌ Table '{table}' not found in database schema")
            available_tables = [t for t in analyzer.tables.keys() if not analyzer.tables[t].is_view]
            if available_tables:
                click.echo(f"💡 Available tables: {', '.join(sorted(available_tables)[:10])}{'...' if len(available_tables) > 10 else ''}")
            return
        
        # Get related tables within specified hops
        related_tables = analyzer.backend.find_related_tables(table, max_distance=hops)
        
        if not related_tables:
            click.echo(f"❌ No related tables found within {hops} hops from '{table}'")
            return
        
        # Separate tables and views
        related_table_names = []
        related_view_names = []
        
        for related_table in related_tables:
            if related_table in analyzer.tables:
                if analyzer.tables[related_table].is_view:
                    related_view_names.append(related_table)
                else:
                    related_table_names.append(related_table)
        
        # Show results
        click.echo(f"\\n📋 Found {len(related_tables)} related entities within {hops} hops:")
        
        if related_table_names:
            click.echo(f"\\n🗃️  Related Tables ({len(related_table_names)}):")
            for i, related_table in enumerate(sorted(related_table_names), 1):
                # Show connection path (limit to user's hop request)
                path = analyzer.find_connection_path(table, related_table, max_hops=hops)
                if path and len(path) > 1:
                    path_str = " → ".join(path)
                    click.echo(f"  {i}. {related_table}")
                    click.echo(f"     Path: {path_str}")
                else:
                    click.echo(f"  {i}. {related_table}")
        
        if related_view_names and show_views:
            click.echo(f"\\n👁️  Related Views ({len(related_view_names)}):")
            for i, related_view in enumerate(sorted(related_view_names), 1):
                # Show connection path (limit to user's hop request)
                path = analyzer.find_connection_path(table, related_view, max_hops=hops)
                if path and len(path) > 1:
                    path_str = " → ".join(path)
                    click.echo(f"  {i}. {related_view}")
                    click.echo(f"     Path: {path_str}")
                else:
                    click.echo(f"  {i}. {related_view}")
        elif related_view_names and not show_views:
            click.echo(f"\\n👁️  Found {len(related_view_names)} related views (use --show-views to show them)")
        
        # Show table details for the source table
        source_table_info = analyzer.tables[table]
        click.echo(f"\\n📊 Source Table Details: {table}")
        click.echo(f"  • Columns: {len(source_table_info.columns)}")
        click.echo(f"  • Foreign Keys: {len(source_table_info.foreign_keys)}")
        if source_table_info.foreign_keys:
            click.echo("  • References:")
            for fk in source_table_info.foreign_keys:
                fk_info = f"    - {', '.join(fk['constrained_columns'])} → {fk['referred_table']}"
                click.echo(fk_info)
        
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', help='Database connection string (overrides DATABASE_URL env var)')
@click.option('--method', '-m', default='community', type=click.Choice(['community', 'importance']), help='Clustering method: community detection or importance-based')
@click.option('--min-size', default=4, help='Minimum cluster size (importance method only)')
@click.option('--max-hops', default=2, help='Maximum relationship hops (importance method only)')
@click.option('--top-pct', default=0.2, help='Percentage of top tables to use as cores (importance method only)')
@backend_options
def create_clusters(connection: str, method: str, min_size: int, max_hops: int, top_pct: float, backend: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, include_views: bool) -> None:
    """Calculate and store table clusters in Neo4j database."""
    try:
        # Use DATABASE_URL from environment if connection not provided
        connection_final = connection or os.getenv('DATABASE_URL')
        if not connection_final:
            click.echo("❌ Database connection required: use -c/--connection or set DATABASE_URL environment variable", err=True)
            sys.exit(1)
        
        # Force Neo4j backend for cluster storage
        if backend != 'neo4j':
            click.echo("⚠️  Cluster storage requires Neo4j backend, switching to Neo4j...")
            backend = 'neo4j'
            
        analyzer = _create_analyzer(connection_final, backend, neo4j_uri, neo4j_user, neo4j_password)
        
        click.echo("🔍 Analyzing database schema...")
        analyzer.analyze_schema(include_views=include_views)
        
        if method == 'importance':
            click.echo(f"🧮 Calculating importance-based clusters (min_size={min_size}, max_hops={max_hops}, top_pct={top_pct:.1%})...")
            clusters = analyzer.backend.find_importance_based_clusters(
                min_cluster_size=min_size,
                max_hops=max_hops,
                top_tables_pct=top_pct
            )
            all_clusters = clusters  # For consistent reporting
        else:
            click.echo("🧮 Calculating community-based table clusters...")
            all_clusters = analyzer.backend.find_table_clusters()
            
            # Filter clusters to minimum size of 4 tables
            clusters = [cluster for cluster in all_clusters if len(cluster) >= min_size]
        
        if not clusters:
            click.echo(f"❌ No clusters with {min_size}+ tables found in the database schema")
            if all_clusters and method == 'community':
                small_clusters = len(all_clusters) - len(clusters)
                click.echo(f"ℹ️  Found {small_clusters} clusters with <{min_size} tables (filtered out)")
            return
        
        click.echo(f"📊 Found {len(clusters)} clusters with {min_size}+ tables")
        if len(all_clusters) > len(clusters) and method == 'community':
            filtered_count = len(all_clusters) - len(clusters)
            click.echo(f"🔽 Filtered out {filtered_count} small clusters (<{min_size} tables)")
        
        # Display cluster information before analysis
        for i, cluster in enumerate(clusters, 1):
            cluster_size = len(cluster)
            cluster_tables = ', '.join(sorted(cluster)[:5])  # Show first 5 tables
            if len(cluster) > 5:
                cluster_tables += f" ... (+{len(cluster) - 5} more)"
            click.echo(f"  Cluster {i}: {cluster_size} tables - {cluster_tables}")
        
        click.echo("🤖 Analyzing clusters with LLM for naming and descriptions...")
        try:
            cluster_analyzer = LLMClusterAnalyzer()
            max_concurrent = int(os.getenv('LLM_MAX_CONCURRENT', '5'))
            
            # Convert clusters to list format for LLM analysis
            cluster_lists = [list(cluster) for cluster in clusters]
            cluster_analyses = cluster_analyzer.analyze_clusters_batch_sync(cluster_lists, max_concurrent)
            
            click.echo("💾 Storing enhanced clusters in Neo4j...")
            analyzer.backend.store_table_clusters_with_analysis(clusters, cluster_analyses)
            
            # Display enhanced cluster information
            click.echo("\n📋 Generated Cluster Analysis:")
            for analysis in cluster_analyses:
                click.echo(f"\n🏷️  {analysis.name} ({analysis.cluster_id})")
                click.echo(f"   📝 {analysis.description}")
                click.echo(f"   🏢 Domain: {analysis.business_domain}")
                if analysis.keywords:
                    click.echo(f"   🏷️  Keywords: {', '.join(analysis.keywords)}")
                click.echo(f"   📊 Confidence: {analysis.confidence:.2f}")
                
        except Exception as e:
            click.echo(f"⚠️  LLM analysis failed ({e}), falling back to basic cluster storage...")
            analyzer.backend.store_table_clusters(clusters)
        
        click.echo("✅ Clusters successfully created and stored!")
        click.echo(f"🏘️  Total clusters: {len(clusters)}")
        click.echo(f"📋 Total tables clustered: {sum(len(cluster) for cluster in clusters)}")
        
        # Show cluster statistics
        cluster_sizes = [len(cluster) for cluster in clusters]
        if cluster_sizes:
            click.echo(f"📏 Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', help='Database connection string (overrides DATABASE_URL env var)')
@click.option('--query', '-q', required=True, help='Natural language query to find relevant tables')
@click.option('--max-tables', default=5, help='Maximum number of tables to return')
@click.option('--max-views', default=5, help='Maximum number of views to return')
@click.option('--combined/--separate', default=False, help='Show combined results or separate tables/views')
@click.option('--neo4j-uri', help='Neo4j connection URI (overrides NEO4J_URI env var)')
@click.option('--neo4j-user', help='Neo4j username (overrides NEO4J_USER env var)')
@click.option('--neo4j-password', help='Neo4j password (overrides NEO4J_PASSWORD env var)')
def find_tables_semantic(connection: str, query: str, max_tables: int, max_views: int, combined: bool, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
    """Find relevant tables using semantic keyword matching."""
    try:
        # Use DATABASE_URL from environment if connection not provided
        connection_final = connection or os.getenv('DATABASE_URL')
        if not connection_final:
            click.echo("❌ Database connection required: use -c/--connection or set DATABASE_URL environment variable", err=True)
            sys.exit(1)
            
        # Initialize analyzer with Neo4j backend (required for semantic search)
        analyzer = _create_analyzer(connection_final, 'neo4j', neo4j_uri, neo4j_user, neo4j_password)
        
        click.echo(f"🔍 Searching for tables related to: '{query}'")
        
        # Extract keywords from the user query (simple word splitting)
        search_keywords = [word.strip().lower() for word in query.replace(',', ' ').split() if len(word.strip()) > 2]
        
        if combined:
            # Use original combined search
            results = analyzer.backend.find_tables_by_keywords(search_keywords, max_tables + max_views)
            
            if not results:
                click.echo("❌ No relevant tables found for your query")
                return
            
            click.echo(f"\n📋 Found {len(results)} relevant tables/views:")
            
            for i, result in enumerate(results, 1):
                entity_type = "View" if result['is_view'] else "Table"
                click.echo(f"\n{i}. {result['table_name']} ({entity_type}) - Score: {result['relevance_score']}")
                
                if result['keyword_matches']:
                    click.echo(f"   🏷️  Keyword matches: {', '.join(result['keyword_matches'])}")
                
                if result['concept_matches']:
                    click.echo(f"   🏢 Business concept matches: {', '.join(result['concept_matches'])}")
        else:
            # Use separated search for tables and views
            results = analyzer.backend.find_tables_and_views_by_keywords(search_keywords, max_tables, max_views)
            
            if not results['tables'] and not results['views']:
                click.echo("❌ No relevant tables or views found for your query")
                return
            
            # Display tables
            if results['tables']:
                click.echo(f"\n🗃️  Top {len(results['tables'])} Tables:")
                for i, result in enumerate(results['tables'], 1):
                    click.echo(f"\n{i}. {result['table_name']} (Table) - Score: {result['relevance_score']}")
                    
                    if result['keyword_matches']:
                        click.echo(f"   🏷️  Keyword matches: {', '.join(result['keyword_matches'])}")
                    
                    if result['concept_matches']:
                        click.echo(f"   🏢 Business concept matches: {', '.join(result['concept_matches'])}")
            else:
                click.echo(f"\n🗃️  No tables found matching your query")
            
            # Display views
            if results['views']:
                click.echo(f"\n👁️  Top {len(results['views'])} Views:")
                for i, result in enumerate(results['views'], 1):
                    click.echo(f"\n{i}. {result['table_name']} (View) - Score: {result['relevance_score']}")
                    
                    if result['keyword_matches']:
                        click.echo(f"   🏷️  Keyword matches: {', '.join(result['keyword_matches'])}")
                    
                    if result['concept_matches']:
                        click.echo(f"   🏢 Business concept matches: {', '.join(result['concept_matches'])}")
            else:
                click.echo(f"\n👁️  No views found matching your query")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()