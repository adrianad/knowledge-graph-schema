"""LLM-based cluster naming and analysis for database schema clusters."""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class ClusterAnalysisResult:
    """Result of LLM analysis for a table cluster."""
    cluster_id: str
    name: str
    description: str
    keywords: List[str]
    business_domain: str
    confidence: float


class LLMClusterAnalyzer:
    """Analyze table clusters using LLM to generate names, descriptions, and keywords."""
    
    def __init__(self):
        """Initialize the LLM cluster analyzer."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for LLM cluster analysis")
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration from environment
        self.api_base = os.getenv('OPENAI_API_BASE', 'http://localhost:8000/v1')
        self.api_key = os.getenv('OPENAI_API_KEY', 'dummy-key')
        self.model = os.getenv('LLM_MODEL', 'qwen')
        
        # Initialize async OpenAI client for local/compatible API
        self.client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
        
        self.logger.info(f"Initialized LLM cluster analyzer with base: {self.api_base}, model: {self.model}")
    
    async def analyze_cluster(self, cluster_id: str, table_names: List[str], table_schemas: Dict[str, Any] = None) -> ClusterAnalysisResult:
        """Analyze a cluster of tables to generate name, description, and keywords."""
        
        # Create enhanced prompt with table schema information
        if table_schemas:
            # Build detailed table information with columns
            table_details = []
            for table in sorted(table_names):
                if table in table_schemas:
                    schema = table_schemas[table]
                    columns = [col.get('name', 'unknown') for col in schema.get('columns', [])]
                    column_info = ', '.join(columns[:8])  # Show first 8 columns
                    if len(columns) > 8:
                        column_info += f' ... (+{len(columns) - 8} more)'
                    table_details.append(f"- {table}: {column_info}")
                else:
                    table_details.append(f"- {table}")
            table_list = "\n".join(table_details)
            context_note = "Based on these table names and their columns"
        else:
            # Fallback to table names only
            table_list = "\n".join([f"- {table}" for table in sorted(table_names)])
            context_note = "Based on these table names"
        
        prompt = f"""Analyze this cluster of database tables and provide a comprehensive business analysis:

Tables in cluster:
{table_list}

{context_note}, provide:
1. A short, descriptive name for this cluster (2-4 words) that captures its business purpose
2. A meaningful description of what this cluster represents in business terms (1-2 sentences)
3. 3-5 relevant keywords that describe the business domain and functionality
4. The primary business domain (e.g., "User Management", "Financial Operations", "Inventory Control", "Content Management", "Order Processing")

Guidelines:
- Focus on business purpose and functionality, not technical implementation
- Keywords should be business-relevant terms that would help identify this cluster's purpose
- The description should explain what business processes or data this cluster supports
- Use domain-specific terminology when appropriate

Respond in this exact JSON format:
{{
    "name": "Cluster Name",
    "description": "Clear business description of what this cluster represents and supports.",
    "keywords": ["business_keyword1", "domain_keyword2", "functional_keyword3", "process_keyword4", "data_keyword5"],
    "business_domain": "Primary Business Domain",
    "confidence": 0.85
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a database architect expert. Analyze table clusters to understand their business purpose and domain. Always respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=5000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            try:
                result_data = json.loads(result_text)
                
                return ClusterAnalysisResult(
                    cluster_id=cluster_id,
                    name=result_data.get("name", f"Cluster {cluster_id}"),
                    description=result_data.get("description", "Database table cluster"),
                    keywords=result_data.get("keywords", []),
                    business_domain=result_data.get("business_domain", "Unknown"),
                    confidence=result_data.get("confidence", 0.5)
                )
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON response for cluster {cluster_id}: {e}")
                self.logger.debug(f"Raw response: {result_text}")
                
                # Fallback to basic naming
                return ClusterAnalysisResult(
                    cluster_id=cluster_id,
                    name=f"Cluster {cluster_id}",
                    description=f"Database cluster containing {len(table_names)} related tables",
                    keywords=[],
                    business_domain="Database Operations",
                    confidence=0.2
                )
                
        except Exception as e:
            self.logger.error(f"Error analyzing cluster {cluster_id}: {e}")
            return ClusterAnalysisResult(
                cluster_id=cluster_id,
                name=f"Cluster {cluster_id}",
                description=f"Database cluster containing {len(table_names)} related tables",
                keywords=[],
                business_domain="Database Operations",
                confidence=0.1
            )
    
    async def analyze_clusters_batch(self, clusters: List[List[str]], max_concurrent: int = 5) -> List[ClusterAnalysisResult]:
        """Analyze multiple clusters concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(cluster_id: str, table_names: List[str]) -> ClusterAnalysisResult:
            async with semaphore:
                return await self.analyze_cluster(cluster_id, table_names)
        
        tasks = []
        for i, cluster_tables in enumerate(clusters, 1):
            cluster_id = f"cluster_{i}"
            tasks.append(analyze_with_semaphore(cluster_id, list(cluster_tables)))
        
        self.logger.info(f"Starting analysis of {len(tasks)} clusters with max {max_concurrent} concurrent requests")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to analyze cluster {i+1}: {result}")
                # Create fallback result
                valid_results.append(ClusterAnalysisResult(
                    cluster_id=f"cluster_{i+1}",
                    name=f"Cluster {i+1}",
                    description=f"Database cluster containing {len(clusters[i])} related tables",
                    keywords=[],
                    business_domain="Database Operations",
                    confidence=0.1
                ))
            else:
                valid_results.append(result)
        
        return valid_results
    
    def analyze_clusters_batch_sync(self, clusters: List[List[str]], max_concurrent: int = 5) -> List[ClusterAnalysisResult]:
        """Synchronous wrapper for batch cluster analysis."""
        return asyncio.run(self.analyze_clusters_batch(clusters, max_concurrent))