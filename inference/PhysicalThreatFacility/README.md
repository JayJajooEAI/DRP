# inference/PhysicalThreatFacility: the physical-threat detector

A message in (a post, alert or article), a verdict out: is it a **physical threat**, a danger or disruption to people or places tied to a real location?
No client is involved. It is served by the shared server in `../app.py` (see `../README.md`); this folder holds the code and its own models (`model/`),
and imports nothing from outside `inference/`.

## Use it

```bash
curl -X POST localhost:8000/physicalthreat/predict -H 'Content-Type: application/json' \
     -d '{"text": "Shots fired near 5th Ave and Pine St, Seattle"}'
python ../smoke_test.py            # run from inference/: sends the samples to every incident type
```
Body: `text` (required), `use_luna` (default true). Errors: 400 empty or too long text, 422 missing field.

```json
{"prediction": "threat", "decided_by": "rung2",
 "location": {"found": true, "places": ["5th Ave", "Pine St", "Seattle"], "by": "model"},
 "p_threat": 0.9999, "luna": null, "luna_skipped": null, "timings_ms": {"total": 316}}
```
`decided_by` is `rung1` (no place named, so not a threat), `rung2` (the DistilBERT score was clear), `rung3` (GPT-6 Luna answered the unsure ones), or
`rung2_fallback` (unsure, but Luna was off or unavailable, so the score at or above 0.5 decides).

## The ladder

| Rung | File | What it does |
|---|---|---|
| 1 | `location.py` | Does the message name a location? A fine-tuned MiniLMv2 word tagger; if it finds nothing, place rules (street shapes, "X County", ", TX", Japanese and Chinese place suffixes, a gazetteer of countries, states and big cities). No place means not a threat |
| 2 | `threat.py` | DistilBERT P(threat): at least 0.6 is a threat, below 0.3 is not a threat, in between is unsure |
| 3 | `luna.py` | GPT-6 Luna (prompt v14, strict JSON answer) decides the unsure band. Needs `OPENAI_API_KEY` in `inference/.env`; without it the server still works |
| – | `engine.py`, `settings.py` | Runs the rungs for one message; paths and switches |

Results on the benchmark's test set (2,850 messages), from the development run: 891 messages stop at rung 1, 1,956 are decided by rung 2, 3 go to Luna;
precision about 99.5%, recall about 99.7%. See `../../development/PhysicalThreatFacility/README.md`.

## Notes
- The models were checked against the development pipeline on all 2,850 test messages: same location decision and same rung on all of them, and the same answer on all
  2,847 messages that rungs 1 and 2 decided. P(threat) differs by at most 0.0013 (fp16 rounding), and three extracted place names differ by a word or two.
- Luna is an LLM: on the one borderline test message (`ptf_16719`) its answer changed between runs, so a rung 3 answer is not guaranteed to repeat.
- The weights are not in git; see `model/README.md`.
