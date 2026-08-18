"""
One-shot Label Studio project setup for the Tier 2 junior pool.

Does the four things the Dev 3 brief asks for, idempotently, so it can be re-run
after a `docker compose down -v` without hand-clicking through the UI:

  1. creates (or finds) the project and applies `labeling_config.xml`
  2. connects this service as the project's ML backend
  3. registers the ANNOTATION_UPDATED / ANNOTATION_CREATED webhook against
     Dev 4's gateway
  4. optionally imports tasks from routing_qa or from a fixture, so there is
     something to open

Usage
-----
    python label_studio/setup_project.py                # create/update everything
    python label_studio/setup_project.py --dry-run      # print, change nothing
    python label_studio/setup_project.py --import-mock  # add one fixture task
    python label_studio/setup_project.py --show         # report current state

Needs `LABEL_STUDIO_API_TOKEN`. In the Label Studio UI that is
Account & Settings -> Access Token.

A note on the webhook secret: the plan has Dev 4 verifying
`LABEL_STUDIO_WEBHOOK_SECRET`, but community Label Studio's webhook model has no
HMAC-signing field — it sends custom *headers* instead. This script therefore
sets the secret as a header, which is a shared-secret check rather than a
signature. Flagged as divergence D17; it is Dev 4's contract to accept or
reject, and the README says so plainly rather than letting them discover it from
a 401.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

import httpx

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
LABELING_CONFIG = HERE / "labeling_config.xml"

WEBHOOK_ACTIONS = ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"]


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _ls_safe_host(url: str) -> str:
    """
    Rewrite a compose-internal URL into one Django's URLValidator accepts.

    `http://webhook_gateway:8004` -> `http://host.docker.internal:8004`

    Label Studio fires this webhook from its own container, so the replacement
    has to be reachable from inside Docker. `host.docker.internal` resolves on
    Docker Desktop (Windows/macOS). On Linux, set LS_WEBHOOK_URL explicitly, or
    add `extra_hosts: ["host.docker.internal:host-gateway"]` to the label_studio
    service.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if "_" in host or "." not in host and host not in ("localhost",):
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://host.docker.internal{port}"
    return url.rstrip("/")


def _token_kind(token: str) -> str:
    """
    'legacy', 'refresh', 'access', or 'jwt' — read from the token itself.

    Decoding is unverified and deliberately so: this only picks an auth scheme,
    it is not a security check. Label Studio verifies the signature.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return "legacy"
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, json.JSONDecodeError):
        return "jwt"
    return claims.get("token_type") or "jwt"


class LabelStudio:
    """
    Thin Label Studio API client that works out which auth scheme the token needs.

    Label Studio issues three different things people all call "the API key", and
    they are not interchangeable:

    * **Legacy Token** - an opaque 40-char string. Sent as `Authorization: Token <t>`.
    * **Personal Access Token** - a JWT whose payload says `token_type: refresh`.
      It is NOT accepted directly; it must be exchanged at `/api/token/refresh`
      for a short-lived access token, which is then sent as `Bearer <access>`.
    * **Access token** - a JWT with `token_type: access`. Sent as `Bearer <t>`.

    Getting this wrong produces a bare 401 with no hint about which of the three
    you have, so the scheme is detected rather than assumed.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.scheme = self._authenticate(token)

    def _authenticate(self, token: str) -> str:
        kind = _token_kind(token)

        if kind == "refresh":
            access = self._exchange_refresh_token(token)
            if access:
                self._client.headers["Authorization"] = f"Bearer {access}"
                if self._auth_works():
                    return "Bearer (exchanged from a Personal Access Token)"

        for header, label in (
            (f"Bearer {token}", "Bearer"),
            (f"Token {token}", "Token (legacy)"),
        ):
            self._client.headers["Authorization"] = header
            if self._auth_works():
                return label

        raise SystemExit(
            f"Could not authenticate against {self.base_url} with the supplied token "
            f"(detected type: {kind}).\n\n"
            "Checked, in order: exchange-then-Bearer, Bearer, Token.\n\n"
            "Most likely causes:\n"
            "  * The token was copied from a DIFFERENT Label Studio instance. Tokens are\n"
            "    per-deployment; one from an earlier container or a cloud account will not\n"
            "    work here.\n"
            "  * This instance has no account yet. Open the UI, sign up, then copy the\n"
            "    token from Account & Settings.\n"
            "  * The token expired.\n"
        )

    def _exchange_refresh_token(self, token: str) -> Optional[str]:
        try:
            response = self._client.post("/api/token/refresh", json={"refresh": token})
        except httpx.RequestError:
            return None
        if response.status_code >= 400:
            return None
        return (response.json() or {}).get("access")

    def _auth_works(self) -> bool:
        try:
            return self._client.get("/api/projects/").status_code < 400
        except httpx.RequestError:
            return False

    def _request(self, method: str, path: str, **kwargs) -> Any:
        response = self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise SystemExit(
                f"Label Studio {method} {path} failed with HTTP {response.status_code}:\n"
                f"{response.text[:1000]}"
            )
        if not response.content:
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

    # -- projects -------------------------------------------------------

    def list_projects(self) -> List[Dict[str, Any]]:
        payload = self._request("GET", "/api/projects/")
        # Label Studio returns a bare list on some versions and a paginated
        # object on others.
        if isinstance(payload, dict):
            return payload.get("results", [])
        return payload or []

    def find_project(self, title: str) -> Optional[Dict[str, Any]]:
        for project in self.list_projects():
            if project.get("title") == title:
                return project
        return None

    def create_project(self, title: str, label_config: str) -> Dict[str, Any]:
        return self._request(
            "POST", "/api/projects/",
            json={
                "title": title,
                "label_config": label_config,
                "description": (
                    "Tier 2 interactive serving. Masks are pre-annotated by the "
                    "stochastic SAM2 decoder and are deliberately slightly wrong; "
                    "correcting them is what trains the model."
                ),
            },
        )

    def update_label_config(self, project_id: int, label_config: str) -> Dict[str, Any]:
        return self._request(
            "PATCH", f"/api/projects/{project_id}/", json={"label_config": label_config}
        )

    def import_tasks(self, project_id: int, tasks: List[Dict[str, Any]]) -> Any:
        return self._request("POST", f"/api/projects/{project_id}/import", json=tasks)

    # -- ML backends ----------------------------------------------------

    def list_ml_backends(self, project_id: int) -> List[Dict[str, Any]]:
        payload = self._request("GET", "/api/ml/", params={"project": project_id})
        if isinstance(payload, dict):
            return payload.get("results", [])
        return payload or []

    def create_ml_backend(self, project_id: int, url: str, title: str) -> Dict[str, Any]:
        return self._request(
            "POST", "/api/ml/",
            json={
                "project": project_id,
                "url": url,
                "title": title,
                "is_interactive": False,
            },
        )

    def update_ml_backend(self, backend_id: int, url: str) -> Dict[str, Any]:
        return self._request("PATCH", f"/api/ml/{backend_id}/", json={"url": url})

    # -- webhooks -------------------------------------------------------

    def list_webhooks(self, project_id: int) -> List[Dict[str, Any]]:
        payload = self._request("GET", "/api/webhooks/", params={"project": project_id})
        if isinstance(payload, dict):
            return payload.get("results", [])
        return payload or []

    def create_webhook(self, project_id: int, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        return self._request(
            "POST", "/api/webhooks/",
            json={
                "project": project_id,
                "url": url,
                "send_payload": True,
                "send_for_all_actions": False,
                "actions": WEBHOOK_ACTIONS,
                "headers": headers,
                "is_active": True,
            },
        )

    def update_webhook(self, webhook_id: int, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        return self._request(
            "PATCH", f"/api/webhooks/{webhook_id}/",
            json={
                "url": url,
                "actions": WEBHOOK_ACTIONS,
                "send_for_all_actions": False,
                "send_payload": True,
                "headers": headers,
                "is_active": True,
            },
        )


# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------

def ensure_project(ls: LabelStudio, title: str, config: str, dry_run: bool) -> Dict[str, Any]:
    existing = ls.find_project(title)
    if existing:
        print(f"  project      : found #{existing['id']} {title!r}")
        if existing.get("label_config", "").strip() != config.strip():
            if dry_run:
                print("  labeling cfg : WOULD update (differs from labeling_config.xml)")
            else:
                ls.update_label_config(existing["id"], config)
                print("  labeling cfg : updated from labeling_config.xml")
        else:
            print("  labeling cfg : already matches labeling_config.xml")
        return existing

    if dry_run:
        print(f"  project      : WOULD create {title!r}")
        return {"id": -1, "title": title}

    project = ls.create_project(title, config)
    print(f"  project      : created #{project['id']} {title!r}")
    return project


def ensure_ml_backend(ls: LabelStudio, project_id: int, url: str, dry_run: bool) -> None:
    for backend in ls.list_ml_backends(project_id):
        if backend.get("url", "").rstrip("/") == url.rstrip("/"):
            print(f"  ml backend   : already connected (#{backend['id']} -> {url})")
            return
        if dry_run:
            print(f"  ml backend   : WOULD repoint #{backend['id']} to {url}")
            return
        ls.update_ml_backend(backend["id"], url)
        print(f"  ml backend   : repointed #{backend['id']} -> {url}")
        return

    if dry_run:
        print(f"  ml backend   : WOULD connect {url}")
        return
    backend = ls.create_ml_backend(project_id, url, "serving_ui (stochastic SAM2 decoder)")
    print(f"  ml backend   : connected #{backend['id']} -> {url}")


def ensure_webhook(ls: LabelStudio, project_id: int, url: str, secret: str, dry_run: bool) -> None:
    headers = {"X-Label-Studio-Secret": secret} if secret else {}
    if not secret:
        print(
            "  ! LABEL_STUDIO_WEBHOOK_SECRET is unset. The webhook will be registered "
            "unauthenticated, and Dev 4's gateway is specified to reject it."
        )

    for hook in ls.list_webhooks(project_id):
        if hook.get("url", "").rstrip("/") == url.rstrip("/"):
            if dry_run:
                print(f"  webhook      : WOULD refresh #{hook['id']} -> {url}")
                return
            ls.update_webhook(hook["id"], url, headers)
            print(f"  webhook      : refreshed #{hook['id']} -> {url} {WEBHOOK_ACTIONS}")
            return

    if dry_run:
        print(f"  webhook      : WOULD create -> {url} {WEBHOOK_ACTIONS}")
        return
    hook = ls.create_webhook(project_id, url, headers)
    print(f"  webhook      : created #{hook['id']} -> {url} {WEBHOOK_ACTIONS}")


def import_mock_task(ls: LabelStudio, project_id: int, dry_run: bool) -> None:
    """
    Import one task built from the shared `routing_task.json` fixture.

    Useful before routing_qa exists: it gives Label Studio something to open so
    the ML backend, the wiggle and the instrumentation can all be exercised by
    hand. The `data.task_id` field is the one that matters — without it, Dev 4's
    webhook cannot be tied back to a QueueTask.
    """
    # Prefer the Track 3 fixture: its image_url is a real, reachable photo, so the
    # task actually renders and the dimension probe succeeds. The shared
    # routing_task.json points at cdn.example.com, which resolves nowhere - a
    # task imported from it shows a broken canvas and produces no pre-annotation.
    mocks = REPO_ROOT / "tests" / "mocks"
    fixture = next(
        (p for p in (mocks / "routing_task.dev3.json", mocks / "routing_task.json") if p.exists()),
        None,
    )
    if fixture is None:
        raise SystemExit(f"no routing task fixture found in {mocks}")

    task = json.loads(fixture.read_text(encoding="utf-8"))
    payload = [{
        "image": task["image_url"],
        "task_id": task["task_id"],
        "image_id": task["image_id"],
        "queue": task["queue"],
    }]
    print(f"  fixture      : {fixture.name}")
    if "example.com" in task["image_url"]:
        print(
            "  ! This fixture's image_url points at example.com and will not render. "
            "Use tests/mocks/routing_task.dev3.json, or edit the URL."
        )

    if dry_run:
        print(f"  tasks        : WOULD import 1 task -> {json.dumps(payload[0])}")
        return
    ls.import_tasks(project_id, payload)
    print(f"  tasks        : imported 1 task (task_id={task['task_id']}) -> {task['image_url']}")


def show(ls: LabelStudio, title: str) -> None:
    project = ls.find_project(title)
    if not project:
        print(f"No project titled {title!r}.")
        return
    pid = project["id"]
    print(f"project #{pid} {title!r}")
    print(f"  tasks        : {project.get('task_number', '?')}")
    for backend in ls.list_ml_backends(pid):
        print(f"  ml backend   : #{backend['id']} {backend.get('url')} state={backend.get('state')}")
    for hook in ls.list_webhooks(pid):
        print(f"  webhook      : #{hook['id']} {hook.get('url')} actions={hook.get('actions')} active={hook.get('is_active')}")


# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print what would change, change nothing")
    parser.add_argument("--import-mock", action="store_true", help="also import one task from the fixture")
    parser.add_argument("--show", action="store_true", help="report current project state and exit")
    args = parser.parse_args()

    base_url = env("LABEL_STUDIO_URL", "http://localhost:8080")
    token = env("LABEL_STUDIO_API_TOKEN")
    title = env("LS_PROJECT_TITLE", "RLHF Segmentation - Junior Pool")

    if not token:
        print(
            "LABEL_STUDIO_API_TOKEN is not set.\n"
            "Get it from the Label Studio UI: Account & Settings -> Access Token,\n"
            "then put it in .env and re-run.",
            file=sys.stderr,
        )
        return 2

    if not LABELING_CONFIG.exists():
        print(f"Missing {LABELING_CONFIG}", file=sys.stderr)
        return 2

    ls = LabelStudio(base_url, token)

    if args.show:
        show(ls, title)
        return 0

    ml_backend_url = env("LABEL_STUDIO_ML_BACKEND_URL", "http://serving_ui:8003")
    gateway_url = env("WEBHOOK_GATEWAY_URL", "http://webhook_gateway:8004").rstrip("/")

    # Label Studio validates the webhook URL with Django's URLValidator, which
    # REJECTS a compose service name like `webhook_gateway`: underscores are not
    # legal in a hostname, and a single-label host has no TLD. The rejection is a
    # flat "Enter a valid URL." that says nothing about which of the two it is.
    #
    # So the URL Label Studio stores is configured separately from the one
    # serving_ui uses to forward telemetry. serving_ui talks to the gateway
    # container-to-container and does not care about the underscore; Label Studio
    # does, and needs a host that passes validation.
    webhook_url = env("LS_WEBHOOK_URL") or f"{_ls_safe_host(gateway_url)}/webhooks/label-studio"
    secret = env("LABEL_STUDIO_WEBHOOK_SECRET")

    print(f"Label Studio  : {base_url}")
    print(f"Auth          : {ls.scheme}")
    print(f"{'DRY RUN — nothing will be changed' if args.dry_run else 'Applying configuration'}")

    project = ensure_project(ls, title, LABELING_CONFIG.read_text(encoding="utf-8"), args.dry_run)
    project_id = project["id"]

    if project_id > 0:
        ensure_ml_backend(ls, project_id, ml_backend_url, args.dry_run)
        ensure_webhook(ls, project_id, webhook_url, secret, args.dry_run)
        if args.import_mock:
            import_mock_task(ls, project_id, args.dry_run)
    else:
        print("  (project not created in dry-run mode; skipping dependent steps)")

    print(
        "\nRemaining manual step: the instrumentation script must be loaded into the "
        "Label Studio page, or no effort telemetry is captured at all.\n"
        "See services/serving_ui/README.md -> 'Loading the instrumentation script'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
