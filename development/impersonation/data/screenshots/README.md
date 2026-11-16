# screenshots

4,470 profile screenshots, one PNG per page, 900 x 900 pixels, named `<id>.png` (for example `imp_p1_apple_0.png`).
The six platforms are X, Facebook, Instagram, LinkedIn, Telegram and TikTok. Each page was rendered from a template, so the layout of a platform is the same on every page (the parser relies on that).

The `id` in `train.csv`, `val.csv` and `test.csv` is the file name without `.png`.
The file name contains the profile type, so it hints at the label; the pipeline reads only the image.

A copy is in S3 at `drp-pipeline/impersonation/screenshots/`.
