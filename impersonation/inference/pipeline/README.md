# pipeline: the steps behind `POST /predict`

`engine.py` runs them in this order for one screenshot and one client:

| File | Step |
|---|---|
| `ocr.py` | Reads the text on the screenshot (docTR) |
| `layout_parser.py` | Turns the text lines into fields: platform, name, handle, bio, link, stats, posts |
| `avatar.py` | Crops the profile picture and compares it with the client's logo and executive photos (CLIP vectors from RDS, plus a pixel check) |
| `signals.py` | About 100 numbers from the fields, the avatar scores and the client profile |
| `rules.py` | Hand-written points, scaled to 0 to 1; decides when the score is clear |
| `features.py` | Builds the feature table the frozen LightGBM model expects |
| `luna.py` | Asks Luna (OpenAI Decisions API) about pages the rules and the model leave unsure |
| `clients.py` | Fetches the client's profile, logo, executive photos and stored vectors from RDS (cached for 5 minutes) |
| `settings.py` | Paths and switches; reads the environment and `inference/.env` |
| `engine.py` | Loads the models once, then runs the steps and builds the answer |

These modules are copies of the ones in `../../code/utilities/`, adjusted to take one request at a time and to read client data from RDS. The folder imports nothing from outside `inference/`.
