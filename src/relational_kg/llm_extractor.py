"""LLM-based keyword extraction for database schema elements."""

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

from .database import TableInfo


@dataclass
class KeywordExtractionResult:
    """Result of keyword extraction for a table/view."""
    table_name: str
    keywords: List[str]
    business_concepts: List[str]
    confidence: float


class LLMKeywordExtractor:
    """Extract business keywords from database tables and views using LLM."""
    
    def __init__(self):
        """Initialize the LLM keyword extractor."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for LLM keyword extraction")
        
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
        
        self.logger.info(f"Initialized LLM extractor with base: {self.api_base}, model: {self.model}")
    
    async def extract_keywords_from_table(self, table_info: TableInfo) -> KeywordExtractionResult:
        """Extract keywords and business concepts from a single table/view."""
        
        # Prepare table context for the LLM
        table_context = self._prepare_table_context(table_info)
        
        # Create prompt with /no_think prefix for Qwen optimization
        prompt = self._create_extraction_prompt(table_context, table_info.name, table_info.is_view)
        
        try:
            # Call LLM API using async OpenAI client
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a database analyst extracting business keywords from schema information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the response
            result = self._parse_llm_response(response.choices[0].message.content, table_info.name)
            
            self.logger.info(f"Extracted {len(result.keywords)} keywords for {table_info.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to extract keywords for {table_info.name}: {e}")
            # Return empty result on failure
            return KeywordExtractionResult(
                table_name=table_info.name,
                keywords=[],
                business_concepts=[],
                confidence=0.0
            )
    
    def _prepare_table_context(self, table_info: TableInfo) -> str:
        """Prepare table information as context for the LLM."""
        context_parts = []
        
        # Table/view basic info
        entity_type = "view" if table_info.is_view else "table"
        context_parts.append(f"Database {entity_type}: {table_info.name}")
        
        # Columns with types
        context_parts.append("Columns:")
        for col in table_info.columns:
            col_info = f"  - {col.name} ({col.type})"
            if col.primary_key:
                col_info += " [PRIMARY KEY]"
            if col.foreign_key:
                col_info += f" [REFERENCES {col.foreign_key}]"
            context_parts.append(col_info)
        
        # Foreign key relationships (for tables)
        if not table_info.is_view and table_info.foreign_keys:
            context_parts.append("Foreign Key Relationships:")
            for fk in table_info.foreign_keys:
                fk_info = f"  - {', '.join(fk['constrained_columns'])} → {fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                context_parts.append(fk_info)
        
        # View dependencies (for views)
        if table_info.is_view and table_info.view_dependencies:
            context_parts.append("View Dependencies:")
            for dep in table_info.view_dependencies:
                context_parts.append(f"  - Depends on: {dep}")
        
        return "\n".join(context_parts)
    
    def _create_extraction_prompt(self, table_context: str, table_name: str, is_view: bool) -> str:
        """Create the LLM prompt for keyword extraction."""
        entity_type = "view" if is_view else "table"
        
        prompt = f"""/no_think

Convert these database field names into plain English keywords that a non-technical person would use to search for this information:

{table_context}

Your task: Transform technical database terms into plain language search terms that business users would naturally type when looking for this data.

Examples of good transformations:
- "user_id" → "user", "person", "account"
- "order_date" → "order", "purchase", "date", "when"
- "total_amount" → "total", "cost", "price", "amount"
- "supplier_name" → "supplier", "vendor", "company"

Rules:
1. Extract MAXIMUM 20 keywords total
2. Focus on what non-technical users would search for
3. Include both specific terms and general concepts
4. Convert technical jargon to everyday language
5. Think about business processes this {entity_type} supports
6. Include synonyms that users might type

Extract keywords in two categories:

SEARCH_TERMS: Plain language words users would type (e.g., "customer", "order", "payment")
BUSINESS_CONCEPTS: Higher-level business areas (e.g., "sales", "inventory", "accounting")

Format your response as:
SEARCH_TERMS: term1, term2, term3, term4, term5
BUSINESS_CONCEPTS: concept1, concept2, concept3"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str, table_name: str) -> KeywordExtractionResult:
        """Parse the LLM response into structured keyword results."""
        keywords = []
        business_concepts = []
        confidence = 0.8  # Default confidence
        
        try:
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('SEARCH_TERMS:'):
                    search_terms = line.replace('SEARCH_TERMS:', '').strip()
                    keywords.extend([kw.strip() for kw in search_terms.split(',') if kw.strip()])
                elif line.startswith('BUSINESS_CONCEPTS:'):
                    business_keywords = line.replace('BUSINESS_CONCEPTS:', '').strip()  
                    business_concepts.extend([kw.strip() for kw in business_keywords.split(',') if kw.strip()])
            
            # Clean up keywords - remove empty strings and duplicates
            keywords = list(set([kw.lower() for kw in keywords if kw.strip()]))
            business_concepts = list(set([bc.lower() for bc in business_concepts if bc.strip()]))
            
            # Enforce maximum of 20 keywords total
            total_keywords = keywords + business_concepts
            if len(total_keywords) > 20:
                # Keep the first 15 search terms and 5 business concepts
                keywords = keywords[:15]
                business_concepts = business_concepts[:5]
                self.logger.info(f"Truncated keywords for {table_name} to 20 total (was {len(total_keywords)})")
            
            # Adjust confidence based on number of extracted keywords
            total_count = len(keywords) + len(business_concepts)
            if total_count >= 10:
                confidence = 0.9
            elif total_count >= 5:
                confidence = 0.8
            elif total_count >= 3:
                confidence = 0.7
            else:
                confidence = 0.6
                
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response for {table_name}: {e}")
            # Fallback: extract basic keywords from table name
            keywords = self._extract_fallback_keywords(table_name)
            confidence = 0.3
        
        return KeywordExtractionResult(
            table_name=table_name,
            keywords=keywords,
            business_concepts=business_concepts,
            confidence=confidence
        )
    
    def _extract_fallback_keywords(self, table_name: str) -> List[str]:
        """Extract basic keywords as fallback when LLM fails."""
        # Simple fallback: split table name by underscores/camelCase
        import re
        
        # Split on underscores and camelCase
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', table_name.replace('_', ' '))
        
        # Clean and return unique words
        keywords = [word.lower() for word in words if len(word) > 2]
        return list(set(keywords))
    
    async def extract_keywords_batch(self, tables: Dict[str, TableInfo], 
                                    include_views: bool = True,
                                    max_concurrent: int = 10) -> List[KeywordExtractionResult]:
        """Extract keywords for multiple tables/views using async processing."""
        # Filter tables to process
        tables_to_process = []
        for table_name, table_info in tables.items():
            # Skip views if not requested
            if table_info.is_view and not include_views:
                continue
            tables_to_process.append(table_info)
        
        total_count = len(tables_to_process)
        self.logger.info(f"Starting async keyword extraction for {total_count} entities with max {max_concurrent} concurrent requests")
        
        # Process in batches to avoid overwhelming the API
        results = []
        
        for i in range(0, total_count, max_concurrent):
            batch = tables_to_process[i:i + max_concurrent]
            batch_size = len(batch)
            
            self.logger.info(f"Processing batch {i//max_concurrent + 1} ({batch_size} entities)...")
            
            # Create async tasks for this batch
            tasks = [self.extract_keywords_from_table(table_info) for table_info in batch]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to process {batch[j].name}: {result}")
                    # Create empty result for failed table
                    results.append(KeywordExtractionResult(
                        table_name=batch[j].name,
                        keywords=[],
                        business_concepts=[],
                        confidence=0.0
                    ))
                else:
                    results.append(result)
            
            processed = min(i + max_concurrent, total_count)
            self.logger.info(f"Completed batch {i//max_concurrent + 1} - Total processed: {processed}/{total_count}")
        
        self.logger.info(f"Completed async keyword extraction for {len(results)} entities")
        return results
    
    def extract_keywords_batch_sync(self, tables: Dict[str, TableInfo], 
                                   include_views: bool = True,
                                   max_concurrent: int = 10) -> List[KeywordExtractionResult]:
        """Synchronous wrapper for async batch extraction."""
        return asyncio.run(self.extract_keywords_batch(tables, include_views, max_concurrent))