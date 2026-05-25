"""Upload word-count checkpoints for an experiment to the HuggingFace Hub.

For a given experiment (e.g. ``BabyLM-2026-Strict``) this script will:
  1. Create a public model repo ``BabyLM-community/gpt2-baseline-{experiment}``.
  2. Upload the checkpoint with the largest word count as the main revision.
  3. Upload every M/B-suffixed checkpoint as its own revision, named
     ``chck_<N>M`` (e.g. ``chck_1M``, ``chck_100M``, ``chck_1000M`` for ``1B``).
All revisions (and the main branch) include the config and tokenizer that
match the experiment.
"""

import argparse
import gc
import random
import re
import shutil
import tempfile
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from transformers import GPT2Config, GPT2LMHeadModel


CKPT_RE = re.compile(r"^epoch_(\d+)([MB])$")

# Files we expect in a completed upload (from save_pretrained).
REQUIRED_MODEL_FILES = {"config.json"}
MODEL_WEIGHT_FILES = {"model.safetensors", "pytorch_model.bin"}

# Rate-limit / retry tuning.
DEFAULT_SLEEP_BETWEEN_UPLOADS = 3.0  # seconds between successful uploads
MAX_RETRIES = 6
INITIAL_BACKOFF = 10.0  # seconds


def _is_rate_limit_error(err: Exception) -> bool:
    """Detect HF 429 / rate-limit-ish errors."""
    msg = str(err).lower()
    if "429" in msg or "too many requests" in msg or "rate limit" in msg:
        return True
    status = getattr(getattr(err, "response", None), "status_code", None)
    return status == 429 or (status is not None and 500 <= status < 600)


def with_retries(fn, *, description: str):
    """Run ``fn`` with exponential backoff on rate limits / transient errors."""
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except HfHubHTTPError as e:
            if attempt == MAX_RETRIES or not _is_rate_limit_error(e):
                raise
            jitter = random.uniform(0, backoff * 0.25)
            wait = backoff + jitter
            print(
                f"  [retry] {description}: HF error ({e}); "
                f"sleeping {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})"
            )
            time.sleep(wait)
            backoff *= 2


def revision_is_complete(api: HfApi, repo_id: str, revision: str) -> bool:
    """Return True if ``revision`` already has a full model + tokenizer upload.

    We require ``config.json`` and at least one model weight file
    (``model.safetensors`` or ``pytorch_model.bin``) to be present on the
    given revision. Absence of the revision (or any 404) -> False.
    """
    try:
        files = set(
            api.list_repo_files(
                repo_id=repo_id, revision=revision, repo_type="model"
            )
        )
    except (RevisionNotFoundError, RepositoryNotFoundError, EntryNotFoundError):
        return False
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404 or "404" in str(e) or "not found" in str(e).lower():
            return False
        raise
    if not REQUIRED_MODEL_FILES.issubset(files):
        return False
    if not (files & MODEL_WEIGHT_FILES):
        return False
    return True


def parse_word_count(dir_name: str):
    """Return the word count encoded in a checkpoint dir name, or None."""
    m = CKPT_RE.match(dir_name)
    if not m:
        return None
    value, unit = int(m.group(1)), m.group(2)
    multiplier = 1_000_000 if unit == "M" else 1_000_000_000
    return value * multiplier


def revision_name(dir_name: str) -> str:
    """Map ``epoch_1M`` -> ``chck_1M`` and ``epoch_1B`` -> ``chck_1000M``."""
    m = CKPT_RE.match(dir_name)
    if not m:
        raise ValueError(f"Not a word-count checkpoint: {dir_name}")
    value, unit = int(m.group(1)), m.group(2)
    if unit == "B":
        value *= 1000
    return f"chck_{value}M"


def build_hf_model_dir(
    state_dict_path: Path,
    config_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
) -> None:
    """Materialize an HF-compatible folder (model + config + tokenizer)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GPT2Config.from_pretrained(str(config_dir))
    model = GPT2LMHeadModel(config)
    state_dict = torch.load(
        str(state_dict_path), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.save_pretrained(str(output_dir))

    del model, state_dict
    gc.collect()

    # Copy tokenizer files alongside the model.
    for f in Path(tokenizer_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, output_dir / f.name)


def ensure_branch(api: HfApi, repo_id: str, branch: str) -> None:
    """Create a branch if it does not already exist."""
    try:
        api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)
    except TypeError:
        # Older huggingface_hub: no ``exist_ok`` kwarg.
        try:
            api.create_branch(repo_id=repo_id, branch=branch)
        except HfHubHTTPError as e:
            if "already exists" not in str(e):
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        help="Experiment folder name under experiments/ (e.g. BabyLM-2026-Strict)",
    )
    parser.add_argument("--org", default="BabyLM-community")
    parser.add_argument("--repo-prefix", default="gpt2-baseline")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--tokenizers-dir", default="tokenizers")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repo instead of public (default: public).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the upload folders locally but do not touch the Hub.",
    )
    parser.add_argument(
        "--sleep-between-uploads",
        type=float,
        default=DEFAULT_SLEEP_BETWEEN_UPLOADS,
        help="Seconds to pause after each successful upload (rate-limit cushion).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload revisions even if they already look complete on the Hub.",
    )
    args = parser.parse_args()

    exp = args.experiment
    ckpt_root = Path(args.experiments_dir) / exp / "checkpoints"
    config_dir = Path(args.configs_dir) / exp
    tokenizer_dir = Path(args.tokenizers_dir) / exp

    for p, label in [
        (ckpt_root, "checkpoints"),
        (config_dir, "config"),
        (tokenizer_dir, "tokenizer"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} directory: {p}")

    # Collect word-count checkpoints.
    word_ckpts: dict[str, int] = {}
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir():
            continue
        wc = parse_word_count(d.name)
        if wc is None:
            continue
        state_dict = d / "latest_student.pt"
        if not state_dict.exists():
            print(f"[warn] skipping {d.name}: missing latest_student.pt")
            continue
        word_ckpts[d.name] = wc

    if not word_ckpts:
        raise RuntimeError(
            f"No word-count checkpoints (epoch_<N>M / epoch_<N>B) found in {ckpt_root}"
        )

    max_ckpt = max(word_ckpts, key=word_ckpts.get)
    print(f"Found {len(word_ckpts)} word-count checkpoints.")
    print(f"Main checkpoint: {max_ckpt}  ({word_ckpts[max_ckpt]:,} words)")

    repo_id = f"{args.org}/{args.repo_prefix}-{exp}"
    print(f"Target repo: {repo_id}  (private={args.private})")

    api = HfApi()
    if not args.dry_run:
        create_repo(
            repo_id=repo_id,
            private=args.private,
            repo_type="model",
            exist_ok=True,
        )

    with tempfile.TemporaryDirectory() as tmp_root_str:
        tmp_root = Path(tmp_root_str)

        # --- Main branch: the largest checkpoint ---
        main_already = False
        if not args.dry_run and not args.force:
            main_already = revision_is_complete(api, repo_id, "main")
        if main_already:
            print(
                f"\n[main] already complete on the Hub "
                f"(corresponds to {max_ckpt}); skipping."
            )
        else:
            main_dir = tmp_root / "main"
            print(f"\n[main] building from {max_ckpt} -> {main_dir}")
            build_hf_model_dir(
                ckpt_root / max_ckpt / "latest_student.pt",
                config_dir,
                tokenizer_dir,
                main_dir,
            )
            if not args.dry_run:
                with_retries(
                    lambda: api.upload_folder(
                        folder_path=str(main_dir),
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"Upload main checkpoint ({max_ckpt})",
                    ),
                    description="upload main",
                )
                print(f"[main] uploaded")
                time.sleep(args.sleep_between_uploads)
            shutil.rmtree(main_dir)

        # --- Revisions: one per word-count checkpoint ---
        for name, wc in sorted(word_ckpts.items(), key=lambda kv: kv[1]):
            rev = revision_name(name)

            if not args.dry_run and not args.force:
                if revision_is_complete(api, repo_id, rev):
                    print(
                        f"\n[{rev}] already complete on the Hub "
                        f"(from {name}); skipping."
                    )
                    continue

            rev_dir = tmp_root / rev
            print(f"\n[{rev}] building from {name} ({wc:,} words)")
            build_hf_model_dir(
                ckpt_root / name / "latest_student.pt",
                config_dir,
                tokenizer_dir,
                rev_dir,
            )
            if not args.dry_run:
                with_retries(
                    lambda: ensure_branch(api, repo_id, rev),
                    description=f"ensure branch {rev}",
                )
                with_retries(
                    lambda: api.upload_folder(
                        folder_path=str(rev_dir),
                        repo_id=repo_id,
                        repo_type="model",
                        revision=rev,
                        commit_message=f"Upload {name} as revision {rev}",
                    ),
                    description=f"upload {rev}",
                )
                print(f"[{rev}] uploaded")
                time.sleep(args.sleep_between_uploads)
            shutil.rmtree(rev_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
