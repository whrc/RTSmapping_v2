#!/usr/bin/env bash
# Assemble a push-ready Hugging Face Space repo for the review app.
#
# A Space is just a git repo whose README.md front-matter names the SDK. This
# copies the runtime files out of the project and writes that front-matter, so
# deploying is `git push` and nothing is hand-maintained in two places.
#
# NOTE: Docker Spaces require a paid plan (PRO for a personal account, Team or
# Enterprise for an org) — see post-inference/review_campaign.md §10.
#
# Usage:
#   scripts/build_hf_space.sh /tmp/rts-review-space
#   cd /tmp/rts-review-space && git init && git add -A && git commit -m "review app"
#   git remote add origin https://huggingface.co/spaces/<user>/rts-review
#   git push -u origin main
set -euo pipefail

OUT="${1:?usage: build_hf_space.sh <output-dir>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$OUT/review/static" "$OUT/inference"
cp "$REPO/computing/Dockerfile.review"        "$OUT/Dockerfile"
cp "$REPO/review/requirements.txt"            "$OUT/review/requirements.txt"
cp "$REPO/review/__init__.py" "$REPO/review/app.py" "$REPO/review/store.py" \
                                              "$OUT/review/"
cp "$REPO/review/static/rater.html"           "$OUT/review/static/"
cp "$REPO/inference/__init__.py" "$REPO/inference/claim.py" "$OUT/inference/"

# The front-matter is the whole difference between a repo and a Space.
# app_port must match the port uvicorn binds in the Dockerfile.
cat > "$OUT/README.md" <<'EOF'
---
title: RTS Review Campaign
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Internal RTS candidate review queue
---

# RTS Review Campaign

Internal rating tool for the pan-Arctic South 2025Q3 retrogressive-thaw-slump
candidate inventory. Reviewers are served one batch of polygons at a time and
rate each `rts` / `false` / `unsure`.

**Keep this Space private.** It reads a license-restricted PlanetScope-derived
crop archive from Google Cloud Storage via short-lived signed URLs. No imagery
is stored here, but the app must not be publicly reachable.

Protocol and runbook: `post-inference/review_campaign.md` in the RTSmappingDL
repository. Required secrets are listed there (§10) and in §11.2.
EOF

echo "Space repo assembled at $OUT"
find "$OUT" -type f | sort | sed "s|^$OUT/|  |"
