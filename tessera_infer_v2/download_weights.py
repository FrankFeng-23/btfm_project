#!/usr/bin/env python3
"""Download TESSERA v2 checkpoints from the Hugging Face Hub.

The weights are NOT bundled with this repository. They live under the
`geotessera` organisation on Hugging Face and are fetched on demand:

    https://huggingface.co/geotessera

Examples
--------
    # the recommended default student (21 M params, 128-d Matryoshka)
    python download_weights.py --model medium

    # every student, but not the 8 GB teacher
    python download_weights.py --model all-students

    # the 2 B teacher (8.26 GB — see the README before you do this)
    python download_weights.py --model teacher
"""
import argparse
import os
import sys

REPO_PREFIX = "geotessera/TESSERA-V-2.0-2B-"

# name -> (hf repo suffix, file inside the repo, destination dir, approx size)
MODELS = {
    "nano":    ("N", "ckpt/student_nano.pt",        "student/checkpoints", "4 MB"),
    "small":   ("S", "ckpt/student_small.pt",       "student/checkpoints", "28 MB"),
    "medium":  ("M", "ckpt/student_medium.pt",      "student/checkpoints", "84 MB"),
    "large":   ("L", "ckpt/student_large.pt",       "student/checkpoints", "175 MB"),
    "teacher": ("Teacher", "ckpt/tessera_v2_2B_teacher.pt",
                                                    "teacher/checkpoints", "8.26 GB"),
}
STUDENTS = ["nano", "small", "medium", "large"]

HERE = os.path.dirname(os.path.abspath(__file__))


def download(name: str, token: str = None) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is required:  pip install huggingface_hub")

    suffix, filename, dest_rel, size = MODELS[name]
    repo_id = REPO_PREFIX + suffix
    dest_dir = os.path.join(HERE, dest_rel)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(filename))

    if os.path.exists(dest):
        print(f"[skip] {name}: already present at {dest}")
        return dest

    print(f"[get ] {name}: {repo_id}/{filename}  (~{size})")
    cached = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    # Copy out of the HF cache so the checkpoint sits next to the code.
    import shutil
    shutil.copyfile(cached, dest)
    print(f"[done] {name}: {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="medium",
                    choices=list(MODELS) + ["all-students", "all"],
                    help="which checkpoint(s) to fetch (default: medium)")
    ap.add_argument("--token", default=None,
                    help="Hugging Face token (only needed for gated/private access)")
    args = ap.parse_args()

    if args.model == "all-students":
        names = STUDENTS
    elif args.model == "all":
        names = STUDENTS + ["teacher"]
    else:
        names = [args.model]

    for n in names:
        download(n, token=args.token)


if __name__ == "__main__":
    main()
