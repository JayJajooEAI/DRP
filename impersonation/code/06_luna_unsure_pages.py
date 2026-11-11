"""Step 4 on train and val: Luna (OpenAI Decisions API, gpt-6-luna) answers the pages the rules and the model left unsure.
Reads outputs/work/dev_<split>_stage.csv, writes outputs/work/luna_<split>.json.  Answers are cached; spend is capped (see utilities/luna.py).
Needs OPENAI_API_KEY (environment, or the file named by OPENAI_ENV_FILE).  Usage: python 06_luna_unsure_pages.py train val
"""

import sys, json, pandas as pd
from utilities.config import *

# decide_many sends the pages to Luna (with caching and a spend cap) and returns one answer per page
from utilities.luna import decide_many

# Only train / val are accepted here; test goes through 08_run_test.py
for sp in [a for a in sys.argv[1:] if a in ("train", "val")]:
    # d = stage table from step 05, keep only the pages marked "llm" (unsure after rules + ML)
    d = pd.read_csv(f"{WORK}/dev_{sp}_stage.csv")
    d = d[d.stage == "llm"]
    print(sp, "pages for Luna:", len(d), flush=True)

    # Ask Luna and save its answers (choice, probability, cost, errors) per page id
    res = decide_many(d, sp)
    json.dump(res, open(f"{WORK}/luna_{sp}.json", "w"))

    # Summary: money spent on new calls, answers served from the cache, and failed calls
    print(
        "  done; cost $%.4f; cache hits %d; errors %d"
        % (
            sum(o["cost"] for o in res.values() if not o["cache_hit"]),
            sum(o["cache_hit"] for o in res.values()),
            sum(o["error"] is not None for o in res.values()),
        )
    )
