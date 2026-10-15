# Data contracts

All paths below are configurable and ignored by version control.

## Questions

One JSON object per line:

```json
{
  "question_id": "opaque-id",
  "question": "Natural-language question",
  "start_entities": ["Entity name"],
  "gold_answer_set": ["Gold answer name"]
}
```

## Gold-path records for SFT

Each record extends a question with a sequence of schema-valid operations:

```json
{
  "question_id": "opaque-id",
  "question": "Natural-language question",
  "start_entities": ["Entity name"],
  "gold_answer_set": ["Gold answer name"],
  "gold_operations": [
    {
      "action": "get_tail_relations",
      "entity": "Entity name"
    },
    {
      "action": "get_tail_entities",
      "entity": "Entity name",
      "relation": "relation.id"
    }
  ]
}
```

The Stage 2 builder executes every operation against the configured KG tool
service, inserts returned observations, rejects invalid operations, and
rejects trajectories whose final answers were not observed.

`prepare_gold_paths.py` constructs these records by breadth-first search
through the configured OACR KG server. The KG server wraps the Freebase SPARQL
endpoint and preserves edge direction while keeping Freebase MIDs hidden from
the policy.

## Buffered rollouts for the state model

One rollout per line:

```json
{
  "question_id": "opaque-id",
  "question": "Natural-language question",
  "gold_answer_set": ["Gold answer name"],
  "predicted_answer_list": ["Predicted answer name"],
  "turns": [
    {
      "operation": {
        "action": "get_tail_relations",
        "entity": "Entity name"
      },
      "observation": {
        "relations": ["relation.id"]
      }
    }
  ]
}
```


