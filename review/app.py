"""The review campaign web app — a thin shell over `review.store.ReviewStore`.

Serves the rating UI, four JSON endpoints, and the crop images. Crops are
**streamed from GCS through this app** rather than handed out as signed URLs.
Two reasons, in order of weight:

* Signing needs either a service-account key or ``iam.serviceAccounts.signBlob``
  on the runtime identity. Granting the latter is an IAM policy write on the
  service account, which this project's operators do not hold (see
  `review_campaign.md` §10.3), and a key is a secret to store, rotate and leak.
  Proxying needs neither: plain ``storage.objectViewer`` on the one bucket,
  which is already granted.
* A signed URL is a bearer token — anyone it reaches can fetch the pixels for
  six hours. Proxied crops stay behind the front door, so this
  licence-restricted PlanetScope derivative is gated by the same auth as
  everything else.

Nothing is cached on disk; bytes go straight from the GCS response to the
browser.

Configuration is entirely environment (Cloud Run env vars / Space secrets):

    REVIEW_BUCKET       GCS bucket holding the campaign, e.g. rts-arctic-usw1
    REVIEW_PREFIX       campaign state prefix, e.g. inference/2025q3_south/review
    REVIEW_CROP_PREFIX  crop archive prefix, e.g. .../internal/review_crops
    REVIEW_MANIFEST     gs:// URI of manifest.parquet
    IAP_AUDIENCE        optional; the IAP JWT audience to require
    GCP_SA_KEY          optional; only for a host that cannot attach a service
                        account. On Cloud Run leave it unset.

Run locally:
    uvicorn review.app:app --port 7860

Spec: `post-inference/review_campaign.md` §6, §10.
"""

from __future__ import annotations

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from review.store import ReviewStore

logger = logging.getLogger(__name__)

# Crops are immutable once rendered, so let the browser keep them: a reviewer
# stepping back through a batch then re-displays with no fetch at all.
CROP_CACHE_CONTROL = "private, max-age=86400"
STATIC = Path(__file__).parent / "static"

_store: ReviewStore | None = None
_bucket = None


def _connect():
    """Open the campaign bucket.

    Two credential paths, because the app runs in two places:

    * **Cloud Run** — the attached service account, which needs only
      ``storage.objectViewer`` + ``objectCreator`` on the campaign bucket. No
      key exists anywhere.
    * **A key in ``GCP_SA_KEY``** — the fallback for a host that cannot attach
      an identity (an HF Space, say).
    """
    from google.cloud import storage
    from google.oauth2 import service_account

    key = os.environ.get("GCP_SA_KEY")
    if key:
        creds = service_account.Credentials.from_service_account_info(
            json.loads(key))
        client = storage.Client(credentials=creds, project=creds.project_id)
    else:
        import google.auth

        creds, project = google.auth.default()
        client = storage.Client(credentials=creds, project=project)
    return client.bucket(os.environ["REVIEW_BUCKET"])


def _read_manifest(uri: str) -> pd.DataFrame:
    """Load the manifest, reading gs:// through the bucket we already opened.

    Going through the storage client rather than pandas/gcsfs keeps the app on
    the one credential it already holds, instead of also needing gcsfs and a
    second credential path.
    """
    if not uri.startswith("gs://"):
        return pd.read_parquet(uri)
    _, _, path = uri[5:].partition("/")
    return pd.read_parquet(io.BytesIO(_bucket.blob(path).download_as_bytes()))


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Load the manifest and open the bucket once, at process start."""
    global _store, _bucket
    logging.basicConfig(level=logging.INFO)
    _bucket = _connect()
    manifest = _read_manifest(os.environ["REVIEW_MANIFEST"])
    _store = ReviewStore(_bucket, manifest, os.environ["REVIEW_PREFIX"],
                         os.environ["REVIEW_CROP_PREFIX"])
    logger.info("campaign loaded: %d items in %d batches",
                len(manifest), len(_store.batch_ids))
    yield


app = FastAPI(title="RTS review campaign", lifespan=_lifespan)


def _crop_url(key: str) -> str:
    """The app-relative URL the browser fetches one crop from."""
    return f"/crop/{key}"


@app.get("/crop/{key:path}")
def crop(key: str) -> Response:
    """Stream one crop JPEG out of the bucket.

    The path is a full object key, so it is checked against the crop prefix
    before use: this endpoint must never become a way to read the rest of the
    bucket (verdicts, claims, the manifest) through a crafted path.
    """
    from google.api_core.exceptions import NotFound

    prefix = f"{_store.crop_prefix}/"
    if not key.startswith(prefix) or ".." in key:
        raise HTTPException(status_code=404, detail="not a crop")
    try:
        data = _bucket.blob(key).download_as_bytes()
    except NotFound:
        raise HTTPException(status_code=404, detail=f"no crop {key}") from None
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": CROP_CACHE_CONTROL})


# --------------------------------------------------------------------------
# Reviewer identity
# --------------------------------------------------------------------------
def _reviewer(request: Request, fallback: str | None = None) -> str:
    """Who is rating: the IAP-authenticated identity, else a supplied name.

    Behind IAP the identity is taken from the signed assertion and the client
    **cannot** override it — verdict attribution feeds the inter-rater kappa and
    the product's audit trail, so it should not be a field anyone can type. The
    plain ``X-Goog-Authenticated-User-Email`` header is documented as being for
    compatibility only, so it is used solely to read the address *after* the
    ``x-goog-iap-jwt-assertion`` JWT verifies.

    Off IAP (local development, the offline pack, a host without it) there is no
    assertion, and the caller-supplied name is used as before.

    Raises:
        HTTPException: 401 if an assertion is present but does not verify, and
            403 if no identity can be established at all.
    """
    assertion = request.headers.get("x-goog-iap-jwt-assertion")
    if assertion:
        email = _verify_iap_assertion(assertion)
        if email:
            return email
        raise HTTPException(status_code=401,
                            detail="IAP assertion failed verification")
    if fallback:
        return fallback
    raise HTTPException(status_code=403, detail="no reviewer identity")


def _verify_iap_assertion(assertion: str) -> str | None:
    """Verify an IAP JWT and return its email, or None if it does not verify."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    audience = os.environ.get("IAP_AUDIENCE")
    try:
        info = id_token.verify_token(
            assertion, google_requests.Request(), audience=audience,
            certs_url="https://www.gstatic.com/iap/verify/public_key")
    except Exception as exc:  # noqa: BLE001 - any failure means "not verified"
        logger.warning("IAP assertion rejected: %s", exc)
        return None
    return info.get("email")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
class Submission(BaseModel):
    batch_id: str
    verdicts: dict[int, str]
    reviewer: str | None = None   # ignored when IAP supplies an identity


class Claim(BaseModel):
    batch_id: str
    reviewer: str | None = None   # ignored when IAP supplies an identity


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "rater.html")


@app.get("/api/me")
def me(request: Request, reviewer: str | None = None) -> dict:
    """Who the server thinks is rating, and whether that is authenticated.

    Lets the UI skip its name prompt behind IAP and show the real identity.
    """
    authenticated = bool(request.headers.get("x-goog-iap-jwt-assertion"))
    try:
        return {"reviewer": _reviewer(request, reviewer),
                "authenticated": authenticated}
    except HTTPException:
        if authenticated:
            raise
        return {"reviewer": None, "authenticated": False}


def _items_with_urls(batch_id: str) -> list[dict]:
    items = []
    for it in _store.batch_items(batch_id):
        it = dict(it)
        it["tight_url"] = _crop_url(it.pop("tight_key"))
        it["wide_url"] = _crop_url(it.pop("wide_key"))
        it["tight_plain_url"] = _crop_url(it.pop("tight_plain_key"))
        it["wide_plain_url"] = _crop_url(it.pop("wide_plain_key"))
        items.append(it)
    return items


@app.get("/api/next")
def next_batch(request: Request, reviewer: str | None = None) -> dict:
    """Claim the next batch in queue order and hand back its items."""
    batch_id = _store.claim_next(_reviewer(request, reviewer))
    if batch_id is None:
        return {"batch_id": None, "items": []}
    return {"batch_id": batch_id, "items": _items_with_urls(batch_id)}


@app.get("/api/batch/{batch_id}")
def reopen_batch(batch_id: str) -> dict:
    """Re-serve a held batch (browser reload / resume)."""
    if not _store.has_batch(batch_id):
        raise HTTPException(status_code=404, detail=f"unknown batch {batch_id}")
    if batch_id in _store.done_ids():
        raise HTTPException(status_code=409,
                            detail=f"batch {batch_id} is already submitted")
    return {"batch_id": batch_id, "items": _items_with_urls(batch_id)}


@app.post("/api/heartbeat")
def heartbeat(request: Request, claim: Claim) -> dict:
    _store.heartbeat(_reviewer(request, claim.reviewer), claim.batch_id)
    return {"ok": True}


@app.post("/api/batch")
def submit(request: Request, sub: Submission) -> dict:
    """Persist a completed batch. Idempotent — a retry is accepted, not doubled."""
    try:
        written = _store.submit(_reviewer(request, sub.reviewer), sub.batch_id,
                                sub.verdicts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"written": written}


@app.get("/api/progress")
def progress() -> dict:
    return _store.progress()
