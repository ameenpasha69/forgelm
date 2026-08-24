# Deploying the ForgeLM demonstration

Everything under `deploy/` builds the same Gradio app — the hosted sibling of
`scripts/demo_app.py` — for one of two targets. It differs from the local demo
only in the ways a hosted app has to differ:

- the model is loaded once, at import, and reused for every request;
- requests are queued with concurrency 1, because one model instance on a
  shared CPU cannot serve two generations at the same time;
- it binds `0.0.0.0`, and takes its port from `PORT` (Cloud Run) or
  `GRADIO_SERVER_PORT` (Spaces, local);
- a model-loading failure is captured into the UI rather than killing the
  process on a dead port;
- comparison mode is off — loading the base model too would roughly double
  memory for a contrast the README already reports in numbers.

The caveats are not softened for deployment. The adapter was trained on 171
synthetic tickets, so the UI says it should not be used on real ones.

## Layout

```
deploy/
  app.py               the Gradio app, shared by both targets
  requirements.txt     CPU inference dependencies
  build.py             stages a deployable tree
  cloudrun/            Dockerfile + .dockerignore
  hf_space/            Space README front matter, .gitattributes, push script
```

`build.py` stages a self-contained tree because both targets need the
`forgelm` package and the 45 MB adapter to travel with the app, and
duplicating the weights inside the repository once per target is not worth it.

## Google Cloud Run

Cloud Run needs no local Docker: `--source` uploads the staged tree and builds
it with Cloud Build. You do need the `gcloud` CLI, and a project with billing
enabled (the always-free tier still requires a card on file).

```bash
python deploy/build.py --target cloudrun
```

```bash
gcloud run deploy forgelm --source build/cloudrun --region us-central1 --memory 4Gi --cpu 2 --concurrency 4 --timeout 600 --max-instances 1 --no-allow-unauthenticated
```

### Why those numbers

| Flag | Why |
|---|---|
| `--memory 4Gi` | Measured 1.99 GB resident with the model loaded and serving. 2 GiB is too tight once Gradio and request buffers are counted. |
| `--cpu 2` | Generation is a long chain of small matmuls; the app sets torch's thread count from the visible cores, so the second core is used. |
| `--concurrency 4` | Generation is serialised by the Gradio queue anyway. This caps how many people can be waiting on one instance, not how many run at once. |
| `--timeout 600` | A cold start loads ~2 GB of float32 weights before the first token. The default 300 s is enough in practice; 600 s leaves room on a slow build. |
| `--max-instances 1` | One instance is the whole demo. Without a cap, a burst could scale out to several 4 GiB instances and burn the free tier. |
| `--no-allow-unauthenticated` | Keeps the endpoint private, matching the project's own note that an unauthenticated model endpoint should not be exposed. |

Scale-to-zero is the default, so the service costs nothing while idle. The
always-free tier is 360,000 GiB-seconds and 180,000 vCPU-seconds per month,
which at 4 GiB and 2 vCPU is roughly 25 hours of active time a month.

### Reaching a private service

`--no-allow-unauthenticated` means the browser needs an identity token, so open
a local tunnel instead:

```bash
gcloud run services proxy forgelm --region us-central1 --port 8080
```

Then open `http://localhost:8080`. To grant a specific person direct access
instead:

```bash
gcloud run services add-iam-policy-binding forgelm --region us-central1 --member "user:someone@example.com" --role roles/run.invoker
```

To make it genuinely public — anyone with the URL, no authentication — redeploy
with `--allow-unauthenticated`. Consider what that means first: it is an open,
unmetered inference endpoint.

### Cold starts

The base model is baked into the image at its pinned revision rather than
downloaded on boot, so a cold start pays for the image pull and weight
materialisation but not a 1 GB download. First request after an idle period is
still slow. `--min-instances 1` removes that, and is billed continuously.

## Hugging Face Spaces

Since 2026 Hugging Face gates compute-backed Spaces — Gradio and Docker alike —
behind a paid plan for personal accounts. Only static Spaces are free, and a
static Space cannot run this model. With a PRO subscription:

```bash
python deploy/build.py --target hfspace
```

```bash
hf auth login
```

```bash
python deploy/hf_space/deploy_space.py --repo <your-username>/forgelm --private
```

Note that the Hub username is not necessarily the GitHub one.

## Google Colab (free, temporary)

No account beyond a Google one, no card, no build. `notebooks/forgelm_demo_colab.ipynb`
clones the repository, installs the inference dependencies, and serves this same
app on a public `*.gradio.live` URL:

[Open the demo notebook](https://colab.research.google.com/github/ameenpasha69/forgelm/blob/main/notebooks/forgelm_demo_colab.ipynb)

Two cells, about two minutes. It does **not** train anything — it serves the
committed adapter, unlike `forgelm_colab.ipynb`, which runs the whole
experiment.

The trade-off is that the URL is unauthenticated and dies with the runtime — 72
hours at the outside, sooner if the cell stops or Colab recycles the VM. Good
for showing someone; not a permanent home.

Sharing is opt-in via `FORGELM_SHARE=1`. Without it `app.py` binds locally and
never opens a tunnel, so no environment can publish an endpoint by accident:

```bash
FORGELM_SHARE=1 python app.py
```

Regenerate the notebook after changing the generator:

```bash
python scripts/make_demo_notebook.py
```

## Verifying a build locally

The staged tree runs as-is:

```bash
cd build/cloudrun && PORT=7860 python app.py
```

The three example tickets should return the objects recorded in `DEMO.md`, and
`2 + 2 = ?` should be confidently triaged with an invalid `category` — that is
the project's own diagnostic finding, surfaced rather than hidden.
