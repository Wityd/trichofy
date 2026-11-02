# Deploying the Hair-Type CV API to Render

This service is a FastAPI wrapper around the fastai ResNet18 model in
`models/hair-resnet18-model.pkl`.

## Render Blueprint

The repo root contains `render.yaml`. It defines one Python web service with:

| Setting | Value |
|---|---|
| Root directory | `Hair-Type-Classifier` |
| Python | `3.11.9` |
| Build | `pip install -r requirements-render.txt` |
| Start | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health check | `/` |
| Plan | Starter (2 GB RAM) |

CPU-only PyTorch wheels are used on purpose: the default PyPI build is much
larger and unnecessary for inference.

## Deploy

1. Push this branch and merge the PR into `main`.
2. In Render: **New → Blueprint** → select the `trichofy` repo.
3. Set the optional environment variables:
   - `WEATHER_API_KEY` — OpenWeatherMap key for `GET /weather`
   - `FRONTEND_ORIGIN` — extra CORS origin if the frontend URL changes
4. Apply the blueprint and wait for the build.
5. Verify:

   ```bash
   curl https://<service>.onrender.com/
   curl -X POST -F "file=@Hair-Type-Classifier/examples/4.jpg" \
     https://<service>.onrender.com/predict
   ```

## Frontend

`tricofy-frontend/src/App.jsx` defaults to
`https://trichofy-backend.onrender.com`. Either name the Render service
`trichofy-backend`, or set `VITE_API_URL` to the new service URL and redeploy
the frontend.

## Compatibility constraints

- Keep `fastai==2.7.17`: the pickle was saved by this version. fastai 2.8+
  refuses to unpickle it (`fastcore.transform.Pipeline` was moved).
- Keep `albumentations==1.3.0`: newer versions reject the pickled transform
  pipeline (`save_applied_params`, `always_apply`). `app.py` has compatibility
  shims, but the pin is the reliable path.
- The 45 MB model file must stay committed; there is no external download step.

## Raspberry Pi deployment

The model also runs on a Raspberry Pi 4B (8 GB, Debian 13, aarch64) using the
same `app.py`. The files in `deploy/` are a rootless alternative to systemd:

| File | Purpose |
|---|---|
| `deploy/run-api.sh` | Restart loop that runs uvicorn on port 8000 |
| `deploy/trichofy-hair-api.service` | Optional systemd unit (needs sudo to install) |

### One-time setup on the Pi

Python 3.13 works; install the matched CPU wheels (do **not** install the
default `torch` build, which pulls a multi-GB CUDA stack on aarch64):

```bash
cd Hair-Type-Classifier
python3 -m venv .venv
.venv/bin/pip install "torch==2.9.1" "torchvision==0.24.1"
.venv/bin/pip install "fastai==2.8.12" "albumentations==1.4.24" \
  opencv-python-headless fastapi "uvicorn[standard]" python-multipart \
  python-dotenv requests pillow
```

### Start on boot (no sudo)

```bash
chmod +x deploy/run-api.sh
( crontab -l 2>/dev/null; \
  echo "@reboot $PWD/deploy/run-api.sh >> $PWD/api.log 2>&1" ) | crontab -
```

Then start it once and verify:

```bash
setsid nohup ./deploy/run-api.sh >> api.log 2>&1 < /dev/null &
curl http://127.0.0.1:8000/
curl -X POST -F "file=@examples/1.jpg" http://127.0.0.1:8000/predict
```

### Or install the systemd unit (needs sudo)

Edit the paths in `deploy/trichofy-hair-api.service` if the repo is not at
`/home/witness/Hair-Type-Classifier`, then:

```bash
sudo cp deploy/trichofy-hair-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trichofy-hair-api
systemctl status trichofy-hair-api
```

### Exposing the Pi

The API listens on `0.0.0.0:8000` but the Pi is behind NAT, so this is LAN-only
by default (`http://<pi-lan-ip>:8000`). To use it from the deployed frontend,
put an HTTPS tunnel in front of it (e.g. a Cloudflare named tunnel pointing at
`http://localhost:8000`) and set the frontend's `VITE_API_URL` to that hostname.
