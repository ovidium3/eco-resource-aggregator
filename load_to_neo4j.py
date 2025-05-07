import glob
import json
import pathlib

from neo4j import GraphDatabase


def main():
    """Sets up connection details, cypher template. Loads papers into Neo4j database."""
  # 1) connection details
  BOLT_URL  = "bolt://localhost:7687"     
  USER      = "neo4j"
  PASSWORD  = "Str0ngPass!"

  driver = GraphDatabase.driver(BOLT_URL, auth=(USER, PASSWORD))


  # 2) cypher template (category = file name)
  CYPHER = """
  MERGE (c:Category {name:$cat})
  WITH c
  UNWIND $batch AS paper
    MERGE (p:Paper {id:paper.id})
      SET p += paper
    MERGE (c)-[:HAS_PAPER]->(p);
  """

  # 3) load papers
  with driver.session() as session:
      for path in glob.glob("climate_outputs/*.json"):
          cat = pathlib.Path(path).stem
          session.run(CYPHER, cat=cat, batch=json.load(open(path)))

      # full‑text index (one‑time)
      session.run("""
      CREATE FULLTEXT INDEX paperFT IF NOT EXISTS
      FOR (p:Paper) ON EACH [p.title, p.abstract]
      """)

  print("Papers and index loaded!\n")
  driver.close()


if __name__ == "__main__":
    __main__()