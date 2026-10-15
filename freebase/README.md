# 🗄️ Freebase Deployment for OACR

## Architecture

```text
Freebase N-Triples
  → Virtuoso database
  → SPARQL endpoint                     http://127.0.0.1:8890/sparql
  → OACR name-only KG wrapper           scripts/serve_kg.py
  → four-operation KG-tool interface    http://127.0.0.1:8011
```

The wrapper exposes:

```text
get_tail_relations
get_head_relations
get_tail_entities
get_head_entities
```

Freebase MIDs are used internally for routing. KG observations returned to the
policy contain entity names and relation names.

## 1. Storage requirements

The final Google Freebase dump contains approximately 1.9 billion triples. The
download is approximately 22 GB compressed and 250 GB uncompressed. Virtuoso
also needs space for its database files and temporary loading data.

Recommended layout:

```text
freebase/
└── virtuoso/
    ├── database/
    └── import/
        └── freebase-rdf-latest.nt.gz
```

Create the directories:

```bash
mkdir -p freebase/virtuoso/database
mkdir -p freebase/virtuoso/import
```

The `freebase/virtuoso/` directory is ignored by Git.

## 2. Download the Freebase dump

Download the final N-Triples dump from
[Google Freebase Data Dumps](https://developers.google.com/freebase/data).

Place the compressed file at:

```text
freebase/virtuoso/import/freebase-rdf-latest.nt.gz
```

The `.nt.gz` suffix is important because the Virtuoso bulk loader selects the
RDF parser from the filename.

Check the file before loading:

```bash
ls -lh freebase/virtuoso/import/freebase-rdf-latest.nt.gz
gzip --test freebase/virtuoso/import/freebase-rdf-latest.nt.gz
```

## 3. Install Virtuoso with Docker

Install Docker Engine, then pull the official OpenLink image:

```bash
docker pull openlink/virtuoso-opensource-7:latest
```

Create the container:

```bash
docker run -d \
  --name freebase-virtuoso \
  --publish 1111:1111 \
  --publish 8890:8890 \
  --env DBA_PASSWORD="<SET_A_LOCAL_PASSWORD>" \
  --env VIRT_PARAMETERS_DIRSALLOWED="/import" \
  --volume "$(pwd)/freebase/virtuoso/database:/database" \
  --volume "$(pwd)/freebase/virtuoso/import:/import:ro" \
  openlink/virtuoso-opensource-7:latest
```

Ports:

| Port | Purpose |
| --- | --- |
| `1111` | Virtuoso SQL/ISQL connection |
| `8890` | HTTP and SPARQL endpoint |

Mounted paths:

| Host path | Container path | Purpose |
| --- | --- | --- |
| `freebase/virtuoso/database` | `/database` | Persistent Virtuoso database |
| `freebase/virtuoso/import` | `/import` | Read-only RDF import files |

Check startup:

```bash
docker ps --filter name=freebase-virtuoso
docker logs --tail 100 freebase-virtuoso
```

## 4. Import Freebase into Virtuoso

Open ISQL:

```bash
docker exec -it freebase-virtuoso \
  isql 1111 dba "<SET_A_LOCAL_PASSWORD>"
```

Register the dump:

```sql
ld_dir(
  '/import',
  'freebase-rdf-latest.nt.gz',
  'http://freebase.com'
);
```

Start the bulk loader:

```sql
rdf_loader_run();
```

Check the load table:

```sql
SELECT ll_file, ll_state, ll_error
FROM DB.DBA.LOAD_LIST;
```

The import is complete when:

```text
ll_state = 2
ll_error = NULL
```

Persist the loaded database:

```sql
checkpoint;
```

The final `checkpoint` is required. For a split dump, place all parts in
`freebase/virtuoso/import/` and register them with a pattern:

```sql
ld_dir('/import', 'freebase-*.nt.gz', 'http://freebase.com');
rdf_loader_run();
checkpoint;
```

Official reference:
[OpenLink RDF Bulk Loader](https://vos.openlinksw.com/owiki/wiki/VOS/VirtBulkRDFLoader).

## 5. Verify the SPARQL endpoint

Check that the HTTP endpoint is reachable:

```bash
curl --fail http://127.0.0.1:8890/sparql
```

Run a Freebase label query:

```bash
curl --request POST "http://127.0.0.1:8890/sparql" \
  --header "Accept: application/sparql-results+json" \
  --data-urlencode 'query=
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?entity ?name WHERE {
      ?entity ns:type.object.name ?name .
      FILTER(LANG(?name) = "en")
    }
    LIMIT 1'
```

The response must contain:

```text
results.bindings
```

## 6. Configure the OACR wrapper

The default local configuration is `configs/freebase.yaml`:

```yaml
sparql_url: http://127.0.0.1:8890/sparql
default_graph: null

sparql_username_env: OACR_SPARQL_USERNAME
sparql_password_env: OACR_SPARQL_PASSWORD
sparql_bearer_token_env: OACR_SPARQL_TOKEN

sparql_request_timeout: 30
relation_limit: 1000
entity_fetch_limit: 300
cvt_expand_limit: 300
top_k: 0
per_relation_limit: 0

host: 127.0.0.1
port: 8011
```

Configuration fields:

| Field | Description |
| --- | --- |
| `sparql_url` | Virtuoso or compatible SPARQL endpoint |
| `default_graph` | RDF graph IRI; leave `null` when the endpoint uses a union/default graph |
| `sparql_username_env` | Environment variable containing the SPARQL username |
| `sparql_password_env` | Environment variable containing the SPARQL password |
| `sparql_bearer_token_env` | Environment variable containing a bearer token |
| `sparql_request_timeout` | Timeout for one SPARQL request |
| `relation_limit` | Maximum raw relations fetched per entity and direction |
| `entity_fetch_limit` | Maximum raw values fetched per relation |
| `cvt_expand_limit` | Maximum outgoing values read while expanding one CVT |
| `top_k` | Maximum relations returned to the policy; `0` means unlimited |
| `per_relation_limit` | Maximum entities returned per relation; `0` means unlimited |
| `host` | KG-wrapper bind address |
| `port` | KG-wrapper port |

## 7. Start the OACR KG wrapper

From the repository root:

```bash
source .venv/bin/activate
python scripts/serve_kg.py --config configs/freebase.yaml
```

Keep this process running during:

- gold-path construction;
- SFT trajectory construction;
- buffered rollout collection;
- OACR RL;
- inference.

Check the wrapper:

```bash
curl http://127.0.0.1:8011/health
```

Expected response:

```json
{"ok":true,"backend":"freebase-sparql","sessions":0}
```

## 8. Use an existing or online SPARQL endpoint

No local Virtuoso installation is required when a compatible endpoint already
contains the Freebase RDF graph.

Set the endpoint:

```yaml
sparql_url: https://sparql.example.org/sparql
default_graph: http://freebase.com
```

Use `default_graph: null` if the endpoint does not require an explicit graph.

For authenticated endpoints:

```bash
export OACR_SPARQL_USERNAME="<username>"
export OACR_SPARQL_PASSWORD="<password>"
export OACR_SPARQL_TOKEN="<bearer-token>"
```

Set only the variables required by the endpoint, then start the same wrapper:

```bash
python scripts/serve_kg.py --config configs/freebase.yaml
```

## 9. Stop and restart

Stop Virtuoso:

```bash
docker stop freebase-virtuoso
```

Restart the existing database:

```bash
docker start freebase-virtuoso
```

Inspect logs:

```bash
docker logs --tail 100 freebase-virtuoso
```


