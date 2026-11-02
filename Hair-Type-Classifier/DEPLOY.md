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
