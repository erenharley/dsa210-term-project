"""
Debug script to find correct WB Data360 indicator codes for WGI
Run this first, paste output back to Claude
"""
import requests

# Test 1: List all WGI indicators
print("=" * 60)
print("TEST 1: List indicators in WB_WGI database")
print("=" * 60)
r = requests.get("https://data360api.worldbank.org/data360/indicators?datasetId=WB_WGI", timeout=30)
print(f"Status: {r.status_code}")
data = r.json()
# Print first 20 indicator IDs
if isinstance(data, list):
    for item in data[:20]:
        print(item)
elif isinstance(data, dict):
    items = list(data.items())[:20]
    for k, v in items:
        print(f"  {k}: {v}")

# Test 2: Try fetching Germany with no indicator filter to see what's available
print("\n" + "=" * 60)
print("TEST 2: Fetch any WGI data for Germany")
print("=" * 60)
r2 = requests.get(
    "https://data360api.worldbank.org/data360/data",
    params={'DATABASE_ID': 'WB_WGI', 'REF_AREA': 'DEU'},
    timeout=30
)
print(f"Status: {r2.status_code}")
data2 = r2.json()
print(f"Keys: {list(data2.keys())}")
print(f"Count: {data2.get('count', 'N/A')}")
records = data2.get('value', [])
print(f"Records: {len(records)}")
if records:
    print("First record:")
    print(records[0])
    print("\nAll unique INDICATOR values:")
    indicators = set(r['INDICATOR'] for r in records if 'INDICATOR' in r)
    for ind in sorted(indicators):
        print(f"  {ind}")

# Test 3: Try alternative database ID
print("\n" + "=" * 60)
print("TEST 3: Try WB_WGI_2023 or similar")
print("=" * 60)
for db_id in ['WGI', 'WB_WGI_2023', 'WB_WGI_2022']:
    r3 = requests.get(
        "https://data360api.worldbank.org/data360/data",
        params={'DATABASE_ID': db_id, 'REF_AREA': 'DEU'},
        timeout=15
    )
    print(f"  {db_id}: status={r3.status_code}, count={r3.json().get('count', 'N/A')}")
