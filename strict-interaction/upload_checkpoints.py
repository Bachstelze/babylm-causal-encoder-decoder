"""Upload word-count checkpoints for an interactive-DPO experiment to the HuggingFace Hub.

For a given experiment (e.g. ``train_100m``) this script will:
  1. Create a public model repo ``BabyLM-community/{repo_prefix}-{experiment}``
     (use ``--private`` to make it private instead).
  2. Upload the checkpoint with the largest word count as the main revision.
  3. Upload every word-count checkpoint as its own revision, named
     ``chck_<N>M`` (e.g. ``chck_1M``, ``chck_100M``, ``chck_1000M``).

Two sources of checkpoints are merged:
  - ``checkpoint_<N>M`` folders saved explicitly by training (covers 1M..100M).
  - ``interactive_round_<X>`` folders saved at the end of each round; round
    ``X`` corresponds to ``(X+1)*50`` million words trained, so odd ``X >= 3``
    contribute the 200M..1000M tier on a 100M training run. Word counts that
    are also covered by an explicit ``checkpoint_<N>M`` folder are deduped in
    favor of the explicit folder.

All revisions (and the main branch) include the config and tokenizer that
match the experiment's ``dataset_size`` (read from ``logging/exp_cfg.yaml``).
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
import yaml
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from transformers import GPT2Config, GPT2LMHeadModel


CKPT_RE = re.compile(r"^checkpoint_(\d+)M$")
ROUND_RE = re.compile(r"^interactive_round_(\d+)$")
WORDS_PER_ROUND_M = 50  # end of round X = (X+1) * 50M words trained

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
    """Return True if ``revision`` already has a full model + tokenizer upload."""
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


def revision_name(word_count_m: int) -> str:
    return f"chck_{word_count_m}M"


def collect_checkpoints(ckpt_root: Path) -> dict[int, Path]:
    """Return {word_count_in_M: source_dir} for all uploadable checkpoints.

    Word counts come from ``checkpoint_<N>M`` folders directly, plus
    ``interactive_round_<X>`` folders whose end-of-round word count is a
    multiple of 100M and not already covered by an explicit checkpoint folder.
    """
    word_ckpts: dict[int, Path] = {}

    # Explicit checkpoint_<N>M folders (1M..100M tier).
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir():
            continue
        m = CKPT_RE.match(d.name)
        if not m:
            continue
        if not (d / "latest_student.pt").exists():
            print(f"[warn] skipping {d.name}: missing latest_student.pt")
            continue
        word_ckpts[int(m.group(1))] = d

    # Post-round interactive_round_<X> folders (100M-aligned, >100M tier).
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir():
            continue
        m = ROUND_RE.match(d.name)
        if not m:
            continue
        round_idx = int(m.group(1))
        wc_m = (round_idx + 1) * WORDS_PER_ROUND_M
        if wc_m % 100 != 0:
            continue  # only the 100M-aligned rounds
        if wc_m in word_ckpts:
            continue  # already covered by an explicit checkpoint_<N>M folder
        if not (d / "latest_student.pt").exists():
            print(f"[warn] skipping {d.name}: missing latest_student.pt")
            continue
        word_ckpts[wc_m] = d

    return word_ckpts


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

    for f in Path(tokenizer_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, output_dir / f.name)


def ensure_branch(api: HfApi, repo_id: str, branch: str) -> None:
    try:
        api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)
    except TypeError:
        try:
            api.create_branch(repo_id=repo_id, branch=branch)
        except HfHubHTTPError as e:
            if "already exists" not in str(e):
                raise


def read_dataset_size(exp_dir: Path) -> str:
    cfg_path = exp_dir / "logging" / "exp_cfg.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Cannot determine dataset_size: missing {cfg_path}"
        )
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    if "dataset_size" not in cfg:
        raise KeyError(f"'dataset_size' not found in {cfg_path}")
    return cfg["dataset_size"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        help="Experiment folder name under experiments/ (e.g. train_100m)",
    )
    parser.add_argument("--org", default="BabyLM-community")
    parser.add_argument("--repo-prefix", default="gpt2-interaction-dpo")
    parser.add_argument(
        "--repo-name",
        default=None,
        help="Override the repo name (default: '{repo_prefix}-{experiment}').",
    )
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--tokenizers-dir", default="tokenizers")
    parser.add_argument(
        "--dataset-size",
        default=None,
        help="Override dataset_size (default: read from logging/exp_cfg.yaml).",
    )
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
    exp_dir = Path(args.experiments_dir) / exp
    ckpt_root = exp_dir / "checkpoints"

    dataset_size = args.dataset_size or read_dataset_size(exp_dir)
    config_dir = Path(args.configs_dir) / dataset_size
    tokenizer_dir = Path(args.tokenizers_dir) / dataset_size

    for p, label in [
        (ckpt_root, "checkpoints"),
        (config_dir, "config"),
        (tokenizer_dir, "tokenizer"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} directory: {p}")

    word_ckpts = collect_checkpoints(ckpt_root)
    if not word_ckpts:
        raise RuntimeError(
            f"No uploadable checkpoints found in {ckpt_root}"
        )

    max_wc = max(word_ckpts)
    max_dir = word_ckpts[max_wc]
    print(f"Found {len(word_ckpts)} word-count checkpoints (dataset_size={dataset_size}).")
    print(f"Main checkpoint: {max_dir.name}  ({max_wc:,}M words)")

    repo_name = args.repo_name or f"{args.repo_prefix}-{exp}"
    repo_id = f"{args.org}/{repo_name}"
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
                f"(corresponds to {max_dir.name}); skipping."
            )
        else:
            main_dir = tmp_root / "main"
            print(f"\n[main] building from {max_dir.name} -> {main_dir}")
            build_hf_model_dir(
                max_dir / "latest_student.pt",
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
                        commit_message=f"Upload main checkpoint ({max_dir.name})",
                    ),
                    description="upload main",
                )
                print(f"[main] uploaded")
                time.sleep(args.sleep_between_uploads)
            shutil.rmtree(main_dir)

        # --- Revisions: one per word-count checkpoint ---
        for wc in sorted(word_ckpts):
            src = word_ckpts[wc]
            rev = revision_name(wc)

            if not args.dry_run and not args.force:
                if revision_is_complete(api, repo_id, rev):
                    print(
                        f"\n[{rev}] already complete on the Hub "
                        f"(from {src.name}); skipping."
                    )
                    continue

            rev_dir = tmp_root / rev
            print(f"\n[{rev}] building from {src.name} ({wc:,}M words)")
            build_hf_model_dir(
                src / "latest_student.pt",
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
                        commit_message=f"Upload {src.name} as revision {rev}",
                    ),
                    description=f"upload {rev}",
                )
                print(f"[{rev}] uploaded")
                time.sleep(args.sleep_between_uploads)
            shutil.rmtree(rev_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
