# utilities: shared modules used by the numbered scripts

| File | What it does |
|---|---|
| `config.py` | Paths (`data/`, `outputs/`), and the client reference data read from `data/classes.json` and `data/assets/` (brand, domain, logo, protected executives with photos). `IMPERSONATION_OUT` moves the outputs elsewhere. |
| `layout_parser.py` | Turns the OCR lines of a screenshot into fields: platform, handle, name, bio, link, follower counts, join date, posts, banner text. Rules on positions, not a model. |
| `signals.py` | About 100 numbers per page from the parsed fields, the avatar scores and the client profile (name and handle versus brand, look-alike links, scam / fan / support wording, executive name match, verified-badge detector, account age and counts). |
| `rules.py` | The hand-written point rules, the rescaling of the points to a 0 to 1 score (fixed bounds, not fitted), and the decision (impersonation / not / unsure). |
| `ml_model.py` | The LightGBM model: feature table, settings, fitting, and out-of-fold predictions grouped by customer. |
| `cascade.py` | Ties the stages together: finds the rule thresholds on train, fits the model, applies the frozen pipeline, merges Luna's answers, computes metrics. Also writes and checks the freeze checksums used before the test run. |
| `luna.py` | The Luna step: the written definition of impersonation, the single multiple-choice question, building each request (screenshot, client logo, client facts, automated checks), caching answers and capping spend. |
| `report_sections.py` | The Step 00 (page-by-page enrichment) and Runtime sections of the HTML report. |
| `report_template.html` | The page design used by `10_build_report.py`. |

The code is commented line by line. The scripts that use these modules are in the folder above.
