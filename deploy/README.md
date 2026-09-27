# Deployment

Two supported ways to run GateWatch unsupervised at GCM.

## systemd (bare server, recommended for on-prem)

```bash
sudo mkdir -p /opt/gcm-gatewatch
# copy the project there, then:
sudo cp deploy/gcm-gatewatch.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gcm-gatewatch
journalctl -u gcm-gatewatch -f     # follow logs
```

`Restart=always` brings the process back if it exits, and the capture layer reconnects on
its own when a camera drops. A watchdog agent for softer failures, like a stalled stage, is
planned but not built yet.

## Docker

```bash
docker build -f deploy/Dockerfile -t gcm-gatewatch .
docker run -d --name gcm-gatewatch \
  --env-file .env \
  -v "$PWD/config/config.yaml:/app/config/config.yaml:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v gcm-data:/app/data \
  -p 8080:8080 \
  gcm-gatewatch
```

Model weights and secrets are mounted at run time, never baked into the image.

## Health

`GET /healthz` reports liveness and per-camera last-frame time. Point your uptime
monitor at it.
