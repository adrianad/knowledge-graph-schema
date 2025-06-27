#!/usr/bin/env python3
"""Test view dependency parsing."""

from src.relational_kg.database import DatabaseExtractor

def test_view_parsing():
    """Test parsing of the statisticssupplieryear view."""
    
    # Your view definition
    view_def = """
    SELECT
        CASE
            WHEN (p.year IS NOT NULL) THEN p.year
            ELSE c.year
        END AS year,
        CASE
            WHEN (p.supplierid IS NOT NULL) THEN p.supplierid
            ELSE c.supplierid
        END AS supplierid,
        CASE
            WHEN (p.suppliername IS NOT NULL) THEN p.suppliername
            ELSE c.suppliername
        END AS suppliername,
    p.purchases,
    p.purchasedprice,
    p.currency AS purchasescurrency,
    c.consumables,
    c.consumableprice,
    c.currency AS consumablescurrency
   FROM (public.statisticssupplierpurchases p
     FULL JOIN public.statisticssupplierconsumables c ON (((p.year = c.year) AND (p.supplierid = c.supplierid) AND ((p.currency)::text = (c.currency)::text))))
  ORDER BY p.year, p.supplierid, p.currency
    """
    
    # Mock available tables
    available_tables = [
        'statisticssupplierpurchases',
        'statisticssupplierconsumables', 
        'supplier',
        'container',
        'project'
    ]
    
    # Create extractor and test parsing
    extractor = DatabaseExtractor("dummy://connection")
    dependencies = extractor._parse_view_dependencies(view_def, available_tables)
    
    print("🔍 Testing view dependency parsing...")
    print(f"View definition (simplified): FROM (public.statisticssupplierpurchases p FULL JOIN public.statisticssupplierconsumables c...)")
    print(f"Available tables: {available_tables}")
    print(f"✅ Found dependencies: {dependencies}")
    
    # Expected: ['statisticssupplierpurchases', 'statisticssupplierconsumables']
    expected = ['statisticssupplierpurchases', 'statisticssupplierconsumables']
    
    if set(dependencies) == set(expected):
        print("✅ SUCCESS: Found expected dependencies!")
    else:
        print(f"❌ FAILED: Expected {expected}, got {dependencies}")
        
        # Debug: test individual patterns
        print("\n🔍 Debug - testing individual patterns:")
        import re
        
        view_def_lower = view_def.lower()
        view_def_clean = re.sub(r'\s+', ' ', view_def_lower.strip())
        
        patterns = [
            (r'\bfrom\s+\(?(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)', "Standard FROM"),
            (r'\b(?:inner\s+|left\s+|right\s+|full\s+|cross\s+)?(?:outer\s+)?join\s+(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)', "JOIN"),
            (r'\(\s*(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s+[a-zA-Z_][a-zA-Z0-9_]*', "Parenthetical"),
        ]
        
        for pattern, name in patterns:
            matches = re.findall(pattern, view_def_clean)
            print(f"   {name}: {matches}")

if __name__ == "__main__":
    test_view_parsing()