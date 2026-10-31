#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu VM (Oracle Cloud or any other):
#
#   1. From your computer, copy the secrets next to the code on the server:
#        scp .env config.yaml ubuntu@<server-ip>:~/
#   2. On the server:
#        curl -fsSL https://raw.githubusercontent.com/ShakedN/AI_Secretary/main/deploy/setup.sh -o setup.sh
#        bash setup.sh <server-ip>
#
# Uses <ip>.sslip.io as the domain (free, no signup) unless you pass a real one.
set -euo pipefail

ARG="${1:?usage: bash setup.sh <server-ip or domain>}"
if [[ "$ARG" =~ ^[0-9.]+$ ]]; then DOMAIN="${ARG//./-}.sslip.io"; else DOMAIN="$ARG"; fi
REPO="https://github.com/ShakedN/AI_Secretary.git"
DIR="$HOME/AI_Secretary"

echo "==> Docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi

echo "==> Firewall: open 80/443 (Oracle's Ubuntu images only allow SSH by default)"
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null \
    || sudo iptables -I INPUT 1 -p tcp --dport "$port" -j ACCEPT
done
sudo netfilter-persistent save 2>/dev/null || true

echo "==> Code"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only; else git clone "$REPO" "$DIR"; fi

echo "==> Secrets"
for f in .env config.yaml; do
  if [ -f "$HOME/$f" ]; then mv "$HOME/$f" "$DIR/$f"; fi
  if [ ! -f "$DIR/$f" ]; then echo "Missing $f — copy it from your computer first (see top of this script)."; exit 1; fi
done
chmod 600 "$DIR/.env"
echo "DOMAIN=$DOMAIN" > "$DIR/deploy/.domain.env"

echo "==> Start"
cd "$DIR"
sudo docker compose -f deploy/docker-compose.yml --env-file deploy/.domain.env up -d --build

echo
echo "Done. In a minute or two (HTTPS certificate) this should answer:"
echo "  https://$DOMAIN/health"
echo "Then point the Twilio number at it (from your computer, with PUBLIC_BASE_URL=https://$DOMAIN in .env):"
echo "  .venv/Scripts/python -m scripts.twilio_number secretary"
