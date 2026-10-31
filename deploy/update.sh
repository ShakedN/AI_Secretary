#!/usr/bin/env bash
# Pull the latest code from GitHub and restart. Run on the server:  bash ~/AI_Secretary/deploy/update.sh
set -euo pipefail
cd "$HOME/AI_Secretary"
git pull --ff-only
sudo docker compose -f deploy/docker-compose.yml --env-file deploy/.domain.env up -d --build
sudo docker image prune -f >/dev/null
echo "Updated."
