"""Endpoint tests for review.app — the campaign's HTTP contract.

Network-free: the FakeBucket from `tests/test_claim.py` stands in for GCS, with
IAP JWT verification stubbed. What matters here is the contract the browser
depends on — a claim hands back items with URLs, a reload can re-open a held
batch, a submitted batch 409s on re-open, bad input is a 400 rather than a 500 —
plus the identity rules (behind IAP the authenticated identity wins and the
client cannot override it) and the crop proxy, which must serve crops and refuse
to serve anything else in the bucket.

Spec: `post-inference/review_campaign.md` §6, §10.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import NotFound

from tests.test_claim import _FakeBlob, _FakeBucket
from tests.test_review_store import _manifest


class _BytesBlob(_FakeBlob):
    """Adds the byte read the crop proxy uses (FakeBlob only does text)."""

    def download_as_bytes(self) -> bytes:
        if self.name not in self._store:
            raise NotFound(self.name)
        return self._store[self.name][1]


class _CropBucket(_FakeBucket):
    def blob(self, name: str) -> _BytesBlob:
        return _BytesBlob(self._store, name)


VALID_ASSERTION = "valid-iap-jwt"
IAP_EMAIL = "ada@woodwellclimate.org"


@pytest.fixture
def client(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.parquet"
    _manifest().to_parquet(manifest, index=False)
    monkeypatch.setenv("REVIEW_BUCKET", "fake")
    monkeypatch.setenv("REVIEW_PREFIX", "campaign/review")
    monkeypatch.setenv("REVIEW_CROP_PREFIX", "campaign/crops")
    monkeypatch.setenv("REVIEW_MANIFEST", str(manifest))

    import review.app as app_mod
    monkeypatch.setattr(app_mod, "_connect", lambda: _CropBucket())
    # Stand in for Google's key-server check: one token verifies, others don't.
    monkeypatch.setattr(app_mod, "_verify_iap_assertion",
                        lambda a: IAP_EMAIL if a == VALID_ASSERTION else None)
    with TestClient(app_mod.app) as c:
        yield c


def _iap(assertion: str = VALID_ASSERTION) -> dict:
    return {"x-goog-iap-jwt-assertion": assertion}


def test_claim_returns_items_with_crop_urls(client):
    r = client.get("/api/next", params={"reviewer": "ann"})
    body = r.json()
    assert body["batch_id"] == "b00000"
    assert len(body["items"]) == 4
    first = body["items"][0]
    assert first["tight_url"] == "/crop/campaign/crops/1_t.jpg"
    assert first["wide_url"] == "/crop/campaign/crops/1_w.jpg"
    assert "tight_key" not in first  # the raw key field is replaced, not added


def test_crop_streams_the_jpeg(client):
    import review.app as app_mod
    app_mod._bucket.blob("campaign/crops/1_t.jpg").upload_from_string(b"\xff\xd8jpeg")

    r = client.get("/crop/campaign/crops/1_t.jpg")
    assert r.status_code == 200
    assert r.content == b"\xff\xd8jpeg"
    assert r.headers["content-type"] == "image/jpeg"
    assert "max-age" in r.headers["cache-control"]


def test_crop_refuses_objects_outside_the_crop_prefix(client):
    """The proxy must not become a reader for the rest of the bucket."""
    import review.app as app_mod
    app_mod._bucket.blob("campaign/review/verdicts/ann.jsonl").upload_from_string(
        b"secret")

    for key in ("campaign/review/verdicts/ann.jsonl",
                "campaign/crops/../review/verdicts/ann.jsonl"):
        r = client.get(f"/crop/{key}")
        assert r.status_code == 404, key
        assert b"secret" not in r.content


def test_crop_of_a_missing_object_is_404_not_500(client):
    """20 polygons have no imagery, so absent crops are expected, not a fault."""
    assert client.get("/crop/campaign/crops/99999_t.jpg").status_code == 404


def test_two_reviewers_get_different_batches(client):
    a = client.get("/api/next", params={"reviewer": "ann"}).json()
    b = client.get("/api/next", params={"reviewer": "bob"}).json()
    assert a["batch_id"] != b["batch_id"]


def test_exhausted_campaign_reports_null_batch(client):
    for who in ("ann", "bob", "cat"):
        client.get("/api/next", params={"reviewer": who})
    assert client.get("/api/next", params={"reviewer": "dan"}).json() == \
        {"batch_id": None, "items": []}


def test_reopen_serves_a_held_batch_again(client):
    """The browser-reload path: same items, freshly signed URLs."""
    claimed = client.get("/api/next", params={"reviewer": "ann"}).json()
    again = client.get(f"/api/batch/{claimed['batch_id']}").json()
    assert [i["rts_id"] for i in again["items"]] == \
        [i["rts_id"] for i in claimed["items"]]


def test_reopen_of_a_submitted_batch_is_a_conflict(client):
    claimed = client.get("/api/next", params={"reviewer": "ann"}).json()
    ids = [i["rts_id"] for i in claimed["items"]]
    client.post("/api/batch", json={"reviewer": "ann",
                                    "batch_id": claimed["batch_id"],
                                    "verdicts": {str(i): "rts" for i in ids}})
    assert client.get(f"/api/batch/{claimed['batch_id']}").status_code == 409


def test_reopen_of_an_unknown_batch_is_404(client):
    assert client.get("/api/batch/b99999").status_code == 404


def test_submit_persists_and_is_idempotent(client):
    claimed = client.get("/api/next", params={"reviewer": "ann"}).json()
    payload = {"reviewer": "ann", "batch_id": claimed["batch_id"],
               "verdicts": {str(i["rts_id"]): "rts" for i in claimed["items"]}}
    assert client.post("/api/batch", json=payload).json() == {"written": True}
    assert client.post("/api/batch", json=payload).json() == {"written": False}


def test_bad_verdict_is_a_400_not_a_500(client):
    claimed = client.get("/api/next", params={"reviewer": "ann"}).json()
    verdicts = {str(i["rts_id"]): "rts" for i in claimed["items"]}
    verdicts[str(claimed["items"][0]["rts_id"])] = "maybe"
    r = client.post("/api/batch", json={
        "reviewer": "ann", "batch_id": claimed["batch_id"],
        "verdicts": verdicts})
    assert r.status_code == 400
    assert "invalid verdicts" in r.json()["detail"]


def test_incomplete_batch_is_a_400(client):
    """The UI blocks submit until every item is rated; the API must too."""
    claimed = client.get("/api/next", params={"reviewer": "ann"}).json()
    r = client.post("/api/batch", json={
        "reviewer": "ann", "batch_id": claimed["batch_id"],
        "verdicts": {str(claimed["items"][0]["rts_id"]): "rts"}})
    assert r.status_code == 400
    assert "incomplete" in r.json()["detail"]


def test_progress_tracks_submissions(client):
    assert client.get("/api/progress").json()["items_done"] == 0
    claimed = client.get("/api/next", params={"reviewer": "ann"}).json()
    client.post("/api/batch", json={
        "reviewer": "ann", "batch_id": claimed["batch_id"],
        "verdicts": {str(i["rts_id"]): "rts" for i in claimed["items"]}})
    p = client.get("/api/progress").json()
    assert p["items_done"] == 4
    assert p["headline_max_prob"] == pytest.approx(1.0)


def test_index_serves_the_rater(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "RTS review campaign" in r.text


# --- reviewer identity ----------------------------------------------------
def test_iap_identity_is_used_when_present(client):
    r = client.get("/api/me", headers=_iap())
    assert r.json() == {"reviewer": IAP_EMAIL, "authenticated": True}


def test_client_cannot_override_the_iap_identity(client):
    """Attribution feeds the kappa and the audit trail — it must not be typeable."""
    r = client.get("/api/me", params={"reviewer": "someone-else"},
                   headers=_iap())
    assert r.json()["reviewer"] == IAP_EMAIL


def test_claim_is_recorded_under_the_iap_identity(client):
    claimed = client.get("/api/next", params={"reviewer": "impostor"},
                         headers=_iap()).json()
    ids = [i["rts_id"] for i in claimed["items"]]
    client.post("/api/batch", headers=_iap(),
                json={"batch_id": claimed["batch_id"], "reviewer": "impostor",
                      "verdicts": {str(i): "rts" for i in ids}})
    import review.app as app_mod
    verdicts = app_mod._store.read_verdicts()
    assert set(verdicts["reviewer"]) == {IAP_EMAIL}


def test_an_unverifiable_assertion_is_rejected(client):
    """A forged assertion must 401, not fall through to the typed name."""
    r = client.get("/api/next", params={"reviewer": "ann"},
                   headers=_iap("forged"))
    assert r.status_code == 401


def test_supplied_name_is_used_when_there_is_no_iap(client):
    """Local runs and the offline pack have no assertion and must still work."""
    assert client.get("/api/me", params={"reviewer": "ann"}).json() == \
        {"reviewer": "ann", "authenticated": False}
    assert client.get("/api/next", params={"reviewer": "ann"}).json()["batch_id"]


def test_no_identity_at_all_is_a_403(client):
    assert client.get("/api/next").status_code == 403


def test_me_reports_no_identity_rather_than_failing(client):
    """The UI calls /api/me before it has a name; it needs an answer, not a 403."""
    assert client.get("/api/me").json() == {"reviewer": None,
                                            "authenticated": False}
