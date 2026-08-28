# Roadmap

The plan: which phase lands what, and the design the unbuilt part gets built against.
Everything below "The planned harness" exists on paper only.

The [README](../README.md) carries what runs today; this file carries what is intended.
The split exists because a status section that grows with every phase, next to a plan that does not shrink
in the same step, turns into noise: two thirds of the README described parts that were not there.

Decisions that were actually made are recorded in [DECISIONS.md](DECISIONS.md) instead.

## Phases

Each phase leaves something that runs.

| Phase | Status | Leaves running |
|---|---|---|
| 1 - The service skeleton | ✅ | FastAPI service, pydantic models, pytest suite |
| 2 - The raw LLM layer | ✅ | Structured outputs, tool calling, configuration object, gold questions as typed structure |
| 3 - Retrieval and the harness | next | Manual pipeline, pgvector schema, first diff between two runs |
| 4 - Production concerns | planned | ragas scorer, LangSmith tracing, token budgets, deployment |

### Phase 1 - The service skeleton ✅

A FastAPI service written by hand, with pydantic models and a pytest suite from the first endpoint on.
It exists to have somewhere for the LLM layer to sit, and it is what `/health`, `/version` and the test setup come from.

### Phase 2 - The raw LLM layer ✅

Direct API calls without a framework: messages, system prompts, streaming, structured outputs validated against pydantic, and tool calling.
The configuration object comes first in this phase, ahead of the code that would otherwise hard-wire its values.
Gold questions are defined here as a typed structure, ahead of the database that will hold them.

### Phase 3 - Retrieval and the harness (next)

The pipeline built by hand first: ingestion, chunking, retrieval, generation.
Then pgvector on PostgreSQL, the schema through SQLAlchemy and Alembic, and the first diff between two runs.
LangChain enters as an additional configuration dimension rather than as a rewrite.

### Phase 4 - Production concerns

ragas as a scorer writing into the existing `scores` table, LangSmith tracing for cost per run, token budgets and rate limiting, and deployment.

## The planned harness

### What a run records

| Recorded                                                       | Why it is kept                                                                                          |
|------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| The configuration it ran under                                 | A run I cannot reproduce is a run I cannot compare against.                                             |
| Every retrieved chunk, with score and rank                     | Shows which passage an answer was assembled from, and how close the ones below it came.                 |
| The gold question, its expected answer and its expected source | The expected source is what makes a retrieval metric possible. Without it I can judge the answer alone. |
| The computed scores                                             | The layer the diff is calculated on.                                                                    |

### The configuration dimensions

Four dimensions, deliberately, and one corpus:

| Dimension | What it controls |
|---|---|
| Chunk size and overlap | How the corpus is cut before it is embedded |
| Embedding model | Which vector space the comparison happens in |
| `top_k` and `ef_search` | How many candidates are retrieved, and how hard the index looks for them |
| Prompt version | The instruction the retrieved context is handed to |

### The data model

| Table        | Holds                                                                     |
|--------------|-----------------------------------------------------------------------------|
| `documents`  | The source documents of the corpus.                                       |
| `chunks`     | Cut documents, pointing at the chunking configuration that produced them. |
| `embeddings` | One row per combination of chunk and model.                               |
| `questions`  | Gold questions with expected answer and expected source.                  |
| `runs`       | One row per run, with its configuration frozen at the time it ran.        |
| `retrievals` | Per run and question, the retrieved chunks with score and rank.           |
| `scores`     | The computed metrics the diff reads.                                      |

The diff is a self-join over `scores` on two `run_id`s.

### The corpus

The corpus is the technical documentation of a framework from this project's own stack: FastAPI, LangChain, SQLAlchemy or pgvector.
It is picked for how hard it is to retrieve from, because a corpus that is easy to search flattens the diffs and leaves the tool with little to show.

Three properties make documentation hard in a useful way:

| Property | What makes it hard |
|---|---|
| Near-duplicates | API reference entries are uniform by design and differ only in details |
| Terminology collisions | "client", "session" and "context" mean different things depending on the section |
| Split answers | The explanation and the code example often live in different sections and have to be pulled together |

Staying inside the target stack has a second reason.
Gold questions need an expected answer and an expected source, and I can only write those for a subject I can judge myself.

### Scope

Four configuration dimensions, one corpus, a report as output, no user interface.

Set aside deliberately:

| Set aside | Why |
|---|---|
| A 3D projection of the vector space (t-SNE, UMAP) | Preserves local neighbourhoods but not global distances; cluster sizes and gaps in the picture would be artefacts of the projection |
| Multi-agent orchestration | A single flow holding state across two or three tools covers the pattern this project needs |
| A second corpus | The comparison runs between configurations; a second corpus adds a variable without adding an answer |

## Intentions, not decisions

Each of these concerns a part that is not built.
Until something runs against them they are the direction phase 3 starts in, and they can still be wrong.
Once one of them survives contact with the code, it moves to [DECISIONS.md](DECISIONS.md) with a date on it.

### Why pgvector on PostgreSQL rather than a dedicated vector store

The data this project produces is relational before it is vectorial.
Almost every question it answers is a join: runs against questions, questions against retrievals, retrievals against chunks, chunks against scores, then aggregated per run and compared across two runs.
That is the shape of the workload, and PostgreSQL is built for it.

A dedicated vector store holds vectors with a payload attached.
It retrieves nearest neighbours well, and for a pipeline whose job ends at retrieval that is the better fit.
Here the retrieval result is the input to the analysis rather than the output of the system, so choosing one would mean running a second database next to Postgres and moving the joins into Python.
Two stores to keep consistent, and aggregation code written by hand where SQL already does it.

The volume argument points the same way.
One corpus of framework documentation across a handful of configurations is not a scale at which a specialised engine earns its operational cost.

What it gives up is real: dedicated stores offer richer filtering, hybrid search and sharding out of the box, and pgvector's index tuning is coarser in comparison.
None of those limits bind at this size, and the decision is reversible, because the embeddings are one table.



<!-- TODO Felix: die drei folgenden Begruendungen selbst schreiben. Stichworte sind da, Laenge und Zuschnitt wie bei pgvector oben: erst die eigene Entscheidung, dann was sie kostet. 

### Why one row per chunk and model, not one column per model

> **Noch zu schreiben.**
> Stichworte: eine `vector`-Spalte hat feste Dimension. Verschiedene Embedding-Modelle liefern unterschiedlich lange Vektoren. Die Kombinationszeile ist relational normal und zugleich genau die Struktur, die der Modellvergleich ohnehin braucht. Was sie kostet: mehr Zeilen, ein Join mehr pro Abfrage.

### Why `ef_search` is its own dimension

> **Noch zu schreiben.**
> Stichworte: HNSW und IVFFlat sind approximativ, nicht exakt. Sie liefern nicht garantiert die tatsaechlich naechsten Nachbarn. `ef_search` tauscht Trefferquote gegen Latenz. Analogie aus dem eigenen Stack: ein nicht-abdeckender Index in SQL Server wird zwar benutzt, zieht aber Key Lookups nach sich. Deshalb gehoert der Parameter in den Diff und nicht in die Fussnote.

### What LangChain takes over

> **Noch zu schreiben.**
> Stichworte: Die Pipeline entsteht zuerst von Hand, LangChain kommt danach als zusaetzliche Konfigurationsdimension dazu. Damit wird die Antwort auf "was nimmt das Framework ab" ein Diff aus zwei Laeufen statt einer Behauptung. Framework-Wahl selbst ist strategisch begruendet, nicht fachlich.

-->
