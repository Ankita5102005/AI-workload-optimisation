#!/usr/bin/env bash
# Safe deploy of a repo.tgz (made with `git archive`) into /raid/gpu_profiler/repo.
# Run ON THE SERVER, from /raid/gpu_profiler:
#     bash repo/scripts/deploy_on_server.sh
#
# What it protects against (all of these happened):
#   * a truncated/missing repo.tgz  -> aborts BEFORE touching repo/ (no more wiped repo)
#   * rsync --delete erasing server-only results (controller CSVs, paper/ figures) and the
#     container-trained models -> those paths are excluded from deletion
set -euo pipefail
cd /raid/gpu_profiler
[ -f repo.tgz ] || { echo "ABORT: /raid/gpu_profiler/repo.tgz not found (scp it first)"; exit 1; }
tar -tzf repo.tgz > /dev/null || { echo "ABORT: repo.tgz is corrupt/truncated (re-run scp, let it finish)"; exit 1; }
rm -rf repo_new && mkdir repo_new && tar -xzf repo.tgz -C repo_new
[ -f repo_new/scripts/docker_stage3.sh ] || { echo "ABORT: archive has no scripts/docker_stage3.sh (wrong tarball layout?)"; exit 1; }
mkdir -p repo
rsync -a --delete \
  --exclude 'predictor/model_artifacts/' \
  --exclude 'data/stage6_controller_run*.csv' \
  --exclude 'paper/' \
  repo_new/ repo/
rm -rf repo_new repo.tgz
echo "Deployed OK. Preserved: predictor/model_artifacts/, data/stage6_controller_run*.csv, paper/"
echo "If predictor code or sweep data changed, RETRAIN inside the container before running Stage 6."
