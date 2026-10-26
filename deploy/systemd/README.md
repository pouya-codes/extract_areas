# Running the services with systemd (no containers)

For the current bare-metal GPU host, the same gateway and worker code can run
directly from the conda environment, replacing the `extract-areas-api@N`
replicas one GPU at a time. The unit files below are templates; the instance
name is `<model>:<gpu>`, e.g. `aim-inference-worker@hovernet:1`.

```bash
sudo cp deploy/systemd/aim-inference-gateway.service /etc/systemd/system/
sudo cp deploy/systemd/aim-inference-worker@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aim-inference-gateway
sudo systemctl enable --now aim-inference-worker@deepliif:0 \
                            aim-inference-worker@hovernet:1 \
                            aim-inference-worker@ec_cancer:2 \
                            aim-inference-worker@pdac:3
```

Point AIMViewer at the gateway (`AI_GATEWAY.URL: http://127.0.0.1:8080` in
`config.yaml`) and stop the old replicas with
`sudo systemctl disable --now extract-areas-api.target`.

The worker unit reads `deploy/systemd/worker.env` (gitignored) for secrets
such as `HF_TOKEN` and `GATEWAY_API_TOKEN`; copy `worker.env.example`.
