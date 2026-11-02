#!/usr/bin/env bash
# Supervised runner for the Trichofy hair-type API.
# Restarts uvicorn if it ever exits, so a @reboot cron entry is enough
# to keep the model serving on the Pi without root.
set -u

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR" || exit 1

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

while true; do
  .venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
  echo "$(date -Is) api exited (code $?), restarting in 5s"
  sleep 5
done
