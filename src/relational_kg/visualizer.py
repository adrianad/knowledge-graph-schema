"""Graph visualization components."""

from typing import Dict, List, Any, Optional
import logging

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot

from .graph import SchemaGraph


class GraphVisualizer:
    """Visualize schema knowledge graphs."""
    
    def __init__(self, schema_graph: SchemaGraph):
        """Initialize with a schema graph."""
        self.graph = schema_graph
        self.logger = logging.getLogger(__name__)
    
    def create_matplotlib_plot(
        self, 
        output_file: str = 'schema_graph.png',
        layout_type: str = 'spring',
        figsize: tuple = (12, 8)
    ) -> None:
        """Create static matplotlib visualization."""
        if len(self.graph.graph.nodes) == 0:
            self.logger.warning("Graph is empty, cannot create visualization")
            return
        
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout_type == 'spring':
            pos = nx.spring_layout(self.graph.graph, k=2, iterations=50)
        elif layout_type == 'circular':
            pos = nx.circular_layout(self.graph.graph)
        elif layout_type == 'hierarchical':
            pos = nx.nx_agraph.graphviz_layout(self.graph.graph, prog='dot')
        else:
            pos = nx.spring_layout(self.graph.graph)
        
        # Draw nodes
        node_sizes = [
            self.graph.graph.nodes[node].get('column_count', 1) * 100 
            for node in self.graph.graph.nodes
        ]
        
        nx.draw_networkx_nodes(
            self.graph.graph, pos, 
            node_size=node_sizes,
            node_color='lightblue',
            alpha=0.7
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph.graph, pos,
            edge_color='gray',
            alpha=0.5,
            arrows=True,
            arrowsize=20
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph.graph, pos,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title("Database Schema Knowledge Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Static visualization saved to {output_file}")
    
    def create_interactive_plot(
        self, 
        output_file: str = 'schema_graph.html',
        layout_type: str = 'spring'
    ) -> None:
        """Create interactive Plotly visualization."""
        if len(self.graph.graph.nodes) == 0:
            self.logger.warning("Graph is empty, cannot create visualization")
            return
        
        # Choose layout
        if layout_type == 'spring':
            pos = nx.spring_layout(self.graph.graph, k=2, iterations=50)
        elif layout_type == 'circular':
            pos = nx.circular_layout(self.graph.graph)
        elif layout_type == 'hierarchical':
            try:
                pos = nx.nx_agraph.graphviz_layout(self.graph.graph, prog='dot')
            except:
                pos = nx.spring_layout(self.graph.graph)
        else:
            pos = nx.spring_layout(self.graph.graph)
        
        # Prepare node data
        node_trace = self._create_node_trace(pos)
        edge_trace = self._create_edge_trace(pos)
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text="Database Schema Knowledge Graph",
                    x=0.5,
                    font=dict(size=20)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Node size represents number of columns. Hover for details.",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color='#888', size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        # Save to HTML file
        plot(fig, filename=output_file, auto_open=False)
        self.logger.info(f"Interactive visualization saved to {output_file}")
    
    def _create_node_trace(self, pos: Dict) -> go.Scatter:
        """Create node trace for Plotly visualization."""
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in self.graph.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            node_data = self.graph.graph.nodes[node]
            column_count = node_data.get('column_count', 0)
            columns = node_data.get('columns', [])
            primary_keys = node_data.get('primary_keys', [])
            
            # Hover text
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"Columns: {column_count}<br>"
            if primary_keys:
                hover_text += f"Primary Keys: {', '.join(primary_keys)}<br>"
            hover_text += f"All Columns: {', '.join(columns[:10])}"
            if len(columns) > 10:
                hover_text += f" ... and {len(columns) - 10} more"
            
            node_text.append(hover_text)
            node_size.append(max(10, column_count * 2))
            node_color.append(column_count)
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[node for node in self.graph.graph.nodes()],
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                line=dict(width=2, color='black'),
                opacity=0.8
            )
        )
    
    def _create_edge_trace(self, pos: Dict) -> go.Scatter:
        """Create edge trace for Plotly visualization."""
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge info
            edge_data = self.graph.graph.edges[edge]
            fk_columns = edge_data.get('foreign_key_columns', [])
            strength = edge_data.get('strength', 1.0)
            
            edge_info.append(
                f"{edge[0]} → {edge[1]}<br>"
                f"Foreign Key: {', '.join(fk_columns)}<br>"
                f"Strength: {strength:.2f}"
            )
        
        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
    
    def create_cluster_visualization(
        self, 
        output_file: str = 'schema_clusters.html'
    ) -> None:
        """Create visualization highlighting table clusters."""
        clusters = self.graph.find_table_clusters()
        
        if not clusters:
            self.logger.warning("No clusters found in the graph")
            return
        
        # Assign colors to clusters
        colors = px.colors.qualitative.Set3
        cluster_colors = {}
        
        for i, cluster in enumerate(clusters):
            color = colors[i % len(colors)]
            for table in cluster:
                cluster_colors[table] = color
        
        # Create layout
        pos = nx.spring_layout(self.graph.graph, k=2, iterations=50)
        
        # Create traces for each cluster
        traces = []
        
        for i, cluster in enumerate(clusters):
            if len(cluster) <= 1:
                continue
                
            cluster_nodes = [node for node in cluster if node in self.graph.graph.nodes]
            
            if not cluster_nodes:
                continue
            
            node_x = [pos[node][0] for node in cluster_nodes]
            node_y = [pos[node][1] for node in cluster_nodes]
            
            node_text = []
            node_size = []
            
            for node in cluster_nodes:
                node_data = self.graph.graph.nodes[node]
                column_count = node_data.get('column_count', 0)
                
                hover_text = f"<b>{node}</b><br>Cluster: {i+1}<br>Columns: {column_count}"
                node_text.append(hover_text)
                node_size.append(max(15, column_count * 2))
            
            traces.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=cluster_nodes,
                textposition="middle center",
                textfont=dict(size=10, color='black'),
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    size=node_size,
                    color=colors[i % len(colors)],
                    line=dict(width=2, color='black'),
                    opacity=0.8
                ),
                name=f'Cluster {i+1}'
            ))
        
        # Add edges
        edge_x = []
        edge_y = []
        
        for edge in self.graph.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        traces.append(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#ccc'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Create figure
        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title="Database Schema Clusters",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        plot(fig, filename=output_file, auto_open=False)
        self.logger.info(f"Cluster visualization saved to {output_file}")