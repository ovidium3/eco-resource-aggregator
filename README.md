# Ecological Resource Aggregator

The **Ecological Resource Aggregator** is a tool designed to efficiently retrieve and process climate-related research documents. By constructing a knowledge graph and implementing an intuitive query system, it enables users to easily find relevant ecological resources.

## Table of Contents

- [Dependencies](#dependencies)
- [Precompute Steps](#precompute-steps)
- [Query Time Steps](#query-time-steps)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Datasets](#datasets)

---

## Dependencies

To run the project, ensure the following libraries are installed:

- Python 3.12 or less (Python 3.13 does not work with sentence_transformers)
- Neo4j
- PyTorch (for transformer-based models)
- Requests
- NetworkX
- Numpy and Pandas
- rouge-score for evaluating generated summaries

Alternatively, you can run "pip install -r requirements.txt", which contains a lengthy list of requirements.
Note that not all requirements here may be mandatory.

### Note that OpenAI, CORE and HuggingFace have required API access tokens.

---

## Precompute Steps

Before running the query system, run these scripts to prepare the data:

1. **`prune_climate_kws.py`**: Processes the `Climate-Change-NER` dataset to obtain a pruned set of climate-related keywords.
2. **`get_docs.py`**: Retrieves climate-related documents from the CORE database using the CORE API.
3. **`load_to_neo4j.py`**: Loads the documents into a Neo4j graph database, building an inverted index for fast querying.
4. **`make_sample_queries.py`**: Generates 50 sample queries for each of the 13 climate-related categories to assist in query rewriting.

---

## Query Time Steps

Once precomputation is complete, you can begin querying the system.

1. **Input your query**: The user enters a query related to climate or ecological topics.
2. **Run `app.py`**: Executes the query and begins the process by calling `rewrite_pipeline.py` to rewrite the query.
3. **Query Rewriting**: The query is rewritten to ensure it aligns better with the available data in the knowledge graph.
4. **Search the Knowledge Graph**: The rewritten query is used to search the Neo4j graph database, and relevant documents are returned based on their relevance.

---

## File Descriptions

- **`prune_climate_kws.py`**: Filters and processes the `Climate-Change-NER` dataset.
- **`get_docs.py`**: Retrieves documents from the CORE API.
- **`load_to_neo4j.py`**: Loads documents into Neo4j and builds an inverted index.
- **`MakeSampleQueries.py`**: Generates sample queries for each climate-related category.
- **`rewrite_pipeline.py`**: Rewrites and optimizes the query for better matching with the knowledge graph.
- **`app.py`**: Main script that initiates the querying process.

---

## Usage

### Precompute the Data
Run the following scripts in order to prepare the data:

1. **prune_climate_kws.py**: This script prunes the climate keywords from the Climate-Change-NER dataset.
python prune_climate_kws.py

2. **get_docs.py**: This script retrieves documents from the CORE dataset using the CORE API.
python get_docs.py

3. **shell script**: In a Unix/Linux terminal, run the following command to instantiate the Neo4j database.
mkdir -p neo4j-data
docker run -d --name neo4j \
  --platform linux/arm64/v8 \
  -p 7474:7474 -p 7687:7687 \
  -v "$PWD/neo4j-data":/data \
  -e NEO4J_AUTH=neo4j/Str0ngPass! \
  -e NEO4J_PLUGINS='["bloom"]' \
  -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
  neo4j:5.18-enterprise

4. **load_to_neo4j.py**: This script generates the knowledge graph, preprocesses the documents, and builds an inverted index.
python load_to_neo4j.py

5. **MakeSampleQueries.py**: This script generates 50 sample queries for each of the 13 categories.
python MakeSampleQueries.py

6. **app.py**: After precomputing the data, run the following script to start querying the system:
python app.py

---

## Datasets

We used two datasets in our project, the `Climate-Change-NER` dataset and CORE dataset.
`Climate-Change-NER` is available via HuggingFace, and CORE is available via CORE API or bulk download at core.ac.uk,
although it does require registering an account and waiting for approval to receive an access link.
