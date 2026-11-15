# Gold sets

| File | Owner | Schema | One record per |
|---|---|---|---|
| `g1.yaml` | Roberto | `G1Item` | version-sensitive code task |
| `g2.jsonl` | Lucas | `G2Item` (one JSON object per line) | retrieval query |
| `g3.yaml` | Roberto | `G3Item` | agent decision label |

Gold chunks are `GoldChunkRef(url, heading_path)`, not chunk ids: the id hash includes the chunk's position,
which only exists once the chunker is final.
