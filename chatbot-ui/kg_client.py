from functools import lru_cache
from neo4j import GraphDatabase
import os

BOLT = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER   = os.getenv("NEO4J_USER", "neo4j")
PWD    = os.getenv("NEO4J_PWD",  "Str0ngPass!")

TOP3 = """
CALL db.index.fulltext.queryNodes('paperFT', $q) YIELD node, score
MATCH (node)<-[:HAS_PAPER]-(:Category {name:$cat})
RETURN node.id    AS id,
       node.doi   AS doi,
       node.title AS title,
       score
ORDER BY score DESC
LIMIT 3;
"""

@lru_cache                  # ensure a single shared driver per process
def _driver():
    return GraphDatabase.driver(BOLT, auth=(USER, PWD))

def top_three(category: str, query: str = "") -> list[dict]:
    with _driver().session() as s:
        return [r.data() for r in s.run(TOP3, cat=category, q=query)]