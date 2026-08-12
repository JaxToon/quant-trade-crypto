from graph.neo4j_client import Neo4jClient

db = Neo4jClient.get()

nodes = db.run_read("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt")
edges = db.run_read("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC")

print("=== NODES ===")
for r in nodes:
    print(f"  {r['label']}: {r['cnt']}")

print()
print("=== EDGES ===")
if not edges:
    print("  (none)")
for r in edges:
    print(f"  {r['rel']}: {r['cnt']}")

total = sum(r['cnt'] for r in edges)
print(f"\n  TOTAL EDGES: {total}")
