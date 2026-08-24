"""Create (if needed) and push the staged Space to the Hugging Face Hub.

Authentication is deliberately not handled here. Log in once with

    hf auth login

and this script picks the stored credential up. It never reads a token from
an argument or an environment variable you have to paste, so the token does
not end up in shell history or in a process listing.

    python deploy/build.py --target hfspace
    python deploy/hf_space/deploy_space.py --repo <user>/<space> [--private]

`upload_folder` handles LFS for the adapter weights, and syncs deletions with
`delete_patterns` so a file removed locally does not linger in the Space.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True,
                    help="target Space id, e.g. <hub-username>/forgelm -- note the Hub username need not match the GitHub one")
    ap.add_argument("--folder", default=str(REPO_ROOT / "build" / "hfspace"),
                    help="staged Space directory (see deploy/build.py)")
    ap.add_argument("--private", action="store_true",
                    help="create the Space private (ignored if it exists)")
    ap.add_argument("--message", default="Deploy ForgeLM demonstration Space")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not (folder / "app.py").exists():
        print(f"No app.py under {folder}. Run deploy/build.py first.",
              file=sys.stderr)
        return 1

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()
    try:
        who = api.whoami()
    except Exception:
        print("Not logged in. Run:  hf auth login", file=sys.stderr)
        return 1
    print(f"authenticated as {who.get('name')}")

    try:
        url = api.create_repo(repo_id=args.repo, repo_type="space",
                              space_sdk="gradio", private=args.private,
                              exist_ok=True)
        print(f"space: {url}")
    except HfHubHTTPError as exc:
        print(f"could not create the Space: {exc}", file=sys.stderr)
        return 1

    print(f"uploading {folder} ...")
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo,
        repo_type="space",
        commit_message=args.message,
        # Keep the Space in step with the staged tree rather than accumulating
        # files that no longer exist locally.
        delete_patterns=["*"],
    )
    print(f"\ncommit: {commit.commit_url}")
    print(f"live at: https://huggingface.co/spaces/{args.repo}")
    print("\nThe Space builds on push; watch the Logs tab for the first boot, "
          "which downloads the base model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
