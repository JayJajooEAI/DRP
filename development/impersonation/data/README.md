# data: the benchmark (local copy)

Not in git. A copy of the screenshots and metadata is in S3 (`drp-pipeline/impersonation/`).

| Item | What it is |
|---|---|
| `screenshots/` | 4,470 profile screenshots, 900 x 900 PNG |
| `assets/` | Per-client logo and wordmark, and the AI-generated executive faces |
| `reference/` | Client homepage and login-page screenshots (used only to illustrate the report) |
| `train.csv`, `val.csv`, `test.csv` | One row per screenshot (3,036 / 720 / 714) |
| `classes.json` | The 25 clients (brand, domain, language, split, protected executives) and the 30 profile types ("concepts") |

## The split
By customer: 17 customers in train, 4 in val, 4 in test. No customer is in two splits.

## The lists (`train.csv` and the others)
Each row has the page `id`, the `customer`, `brand` and `official_domain`, the `split`, the profile type (`concept`, `tier`), the truth (`label`: 1 = impersonation, 0 = not; `expected_incident_type`) and the text and numbers shown on the page (platform, display name, handle, bio, link, followers, posts, account age, ...).

**The pipeline reads only `id` and `customer`** (to pick the client profile) and the screenshot. Every other column is ground truth, used only to score the result.

## Profile types
`classes.json` defines 30 types in four tiers: `positive` and `hard_positive` (impersonation, for example a fake brand-support page), `negative` and `hard_negative` (not impersonation, for example a fan page, a critic account, a scam page, a partner or a genuine account).

## Things to know
- The screenshot **file names contain the profile type** (for example `imp_p1_apple_0.png`), so a name hints at the label. The pipeline never uses the name.
- Everything is generated: templated wording, AI-generated executive faces, fictional executives. It is not real social-media data.
