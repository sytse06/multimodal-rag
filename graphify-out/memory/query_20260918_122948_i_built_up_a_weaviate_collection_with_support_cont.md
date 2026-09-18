---
type: "query"
date: "2026-09-18T12:29:48.995578+00:00"
question: "I built up a weaviate collection with support content. It is running locally on docker desktop. What is the best way to share this?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["WeaviateStore", "SupportChunk", "weaviate_data Volume", "Weaviate Service"]
---

# Q: I built up a weaviate collection with support content. It is running locally on docker desktop. What is the best way to share this?

## Answer

Expanded from original query via graph vocab: [weaviate, collection, volume, data, store, support, chunk]. Recommendation: for a reproducible colleague seed, use Weaviate's native backup-filesystem module to create a collection-scoped backup of SupportChunk, bind-mount a host backup directory, package the completed backup with a checksum and compatibility manifest, and share it through approved encrypted artifact or object storage rather than Git. Recipients run the same single-node topology and restore the backup into an instance where SupportChunk does not already exist. The manifest must record Weaviate image version, Python client version, collection name, object count, embedding provider/model and vector dimension, source snapshot date, git commit, and checksum. The query embedding model must match the model that produced the stored vectors. Do not copy the live Docker named volume and do not use JSON export/import unless transforming schema or crossing an incompatible migration. For continuously shared mutable data, use one secured remote Weaviate deployment instead of circulating snapshots. Before any upgrade, back up version 1.28.4; it is outside Weaviate's currently supported minor range. Also bind ports to 127.0.0.1 and stop using anonymous network exposure for a team-facing setup.

## Outcome

- Signal: useful

## Source Nodes

- WeaviateStore
- SupportChunk
- weaviate_data Volume
- Weaviate Service