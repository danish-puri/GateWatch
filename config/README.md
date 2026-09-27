# Configuration

Two layers, on purpose:

- **`config.yaml`** (copy from `config.example.yaml`) — non-secret settings: camera gate lines, model paths, thresholds, agent models, operating hours. Safe to read and review. `config.yaml` itself is gitignored so local tweaks stay local; the committed reference is `config.example.yaml`.
- **`.env`** (in the project root, copy from `.env.example`) — secrets only: the Anthropic API key and the RTSP URLs with camera credentials. Never committed.

The YAML refers to secrets by env-var **name** (e.g. `rtsp_env: GCM_ENTRY_CAM_RTSP`) rather than holding the value. `gcm_gatewatch.config` resolves the name against the environment at load time. This is the fix for v1, which hardcoded RTSP credentials directly in `run.py`.
