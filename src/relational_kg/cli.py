"""Command-line interface for the relational knowledge graph."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from .analyzer import SchemaAnalyzer
from .visualizer import GraphVisualizer


# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose: bool) -> None:
    """Relational Knowledge Graph CLI for database schema analysis."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.option('--connection', '-c', required=True, help='Database connection string')
@click.option('--output', '-o', help='Output file for graph data (JSON)')
def analyze(connection: str, output: Optional[str]) -> None:
    """Analyze database schema and build knowledge graph."""
    try:
        analyzer = SchemaAnalyzer(connection)
        click.echo("Analyzing database schema...")
        
        analyzer.analyze_schema()
        
        # Get summary
        summary = analyzer.get_schema_summary()
        
        click.echo(f"✅ Analysis complete!")
        click.echo(f"📊 Total tables: {summary['total_tables']}")
        click.echo(f"🔗 Graph edges: {summary['graph_statistics']['edge_count']}")
        click.echo(f"🏘️  Table clusters: {len(summary['table_clusters'])}")
        
        # Show most important tables
        click.echo("\\n🌟 Most important tables:")
        for table_info in summary['most_important_tables'][:5]:
            click.echo(f"  • {table_info['table']} (score: {table_info['importance_score']:.3f})")
        
        # Save graph data if requested
        if output:
            graph_data = analyzer.graph.export_graph_data()
            with open(output, 'w') as f:
                json.dump(graph_data, f, indent=2)
            click.echo(f"💾 Graph data saved to {output}")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', required=True, help='Database connection string')
@click.option('--keywords', '-k', required=True, help='Comma-separated keywords to search for')
@click.option('--max-tables', '-m', default=10, help='Maximum number of tables to return')
@click.option('--include-related', '-r', is_flag=True, help='Include related tables in results')
def find_tables(connection: str, keywords: str, max_tables: int, include_related: bool) -> None:
    """Find relevant tables based on keywords."""
    try:
        analyzer = SchemaAnalyzer(connection)
        analyzer.analyze_schema()
        
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
        
        # Export schema subset for all identified tables
        if all_tables:
            schema_subset = analyzer.export_schema_subset(list(all_tables))
            output_file = f"schema_subset_{'-'.join(keyword_list)}.json"
            with open(output_file, 'w') as f:
                json.dump(schema_subset, f, indent=2)
            click.echo(f"\\n💾 Schema subset saved to {output_file}")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', required=True, help='Database connection string')
@click.option('--tables', '-t', required=True, help='Comma-separated list of base tables')
@click.option('--max-suggestions', '-m', default=5, help='Maximum number of suggestions')
def suggest_joins(connection: str, tables: str, max_suggestions: int) -> None:
    """Suggest tables that could be joined with the given base tables."""
    try:
        analyzer = SchemaAnalyzer(connection)
        analyzer.analyze_schema()
        
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
@click.option('--connection', '-c', required=True, help='Database connection string')
@click.option('--output', '-o', default='schema_graph.html', help='Output HTML file')
@click.option('--layout', '-l', default='spring', help='Graph layout (spring, circular, hierarchical)')
def visualize(connection: str, output: str, layout: str) -> None:
    """Generate interactive visualization of the schema graph."""
    try:
        analyzer = SchemaAnalyzer(connection)
        analyzer.analyze_schema()
        
        click.echo("🎨 Generating visualization...")
        
        visualizer = GraphVisualizer(analyzer.graph)
        visualizer.create_interactive_plot(output, layout_type=layout)
        
        click.echo(f"✅ Visualization saved to {output}")
        click.echo(f"🌐 Open {output} in your browser to view the interactive graph")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--connection', '-c', required=True, help='Database connection string')
def summary(connection: str) -> None:
    """Display summary statistics of the database schema."""
    try:
        analyzer = SchemaAnalyzer(connection)
        analyzer.analyze_schema()
        
        summary = analyzer.get_schema_summary()
        
        click.echo("📊 Database Schema Summary")
        click.echo("=" * 40)
        click.echo(f"Database Type: {summary['database_type']}")
        click.echo(f"Total Tables: {summary['total_tables']}")
        
        stats = summary['graph_statistics']
        click.echo(f"Graph Edges: {stats['edge_count']}")
        click.echo(f"Graph Density: {stats['density']:.3f}")
        click.echo(f"Is Connected: {stats['is_connected']}")
        click.echo(f"Connected Components: {stats['strongly_connected_components']}")
        
        click.echo(f"\\n🏘️  Table Clusters ({len(summary['table_clusters'])}):")
        for i, cluster in enumerate(summary['table_clusters'], 1):
            if len(cluster) > 1:  # Only show clusters with multiple tables
                click.echo(f"  {i}. {', '.join(sorted(cluster))}")
        
        click.echo("\\n🌟 Most Important Tables:")
        for table_info in summary['most_important_tables'][:10]:
            click.echo(f"  • {table_info['table']} (score: {table_info['importance_score']:.3f})")
        
        analyzer.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()