# pi-unsloth

Drive [Unsloth Studio](https://github.com/unslothai/unsloth) from inside [Pi](https://pi.dev) — the reverse of `unsloth start pi`.

`unsloth start pi` launches Pi from Unsloth with one static model. This plugin lets Pi drive Unsloth instead:

- **List every model available locally** on the Unsloth server (`/unsloth list`), loaded or not, with quant and context info.
- **Load and unload models** from Pi (`/unsloth load <model>`, `/unsloth unload <model>`, or the interactive `/unsloth` picker), including GGUF quant variant selection.
- Loaded models appear in **`/model`** as native models of the `unsloth` provider, with real context windows and reasoning/vision support when the server reports them.

No model-management logic is reimplemented: the plugin talks to the running Studio backend's own REST API and reuses `unsloth_cli`'s Python helpers for server discovery, identity verification, and token minting.

## Install

```bash
pi install git:github.com/dangayle/pi-unsloth@v1
```

Or try it for one session:

```bash
pi --extension ./extensions/index.ts
```

## Requirements

- Unsloth Studio installed and running (the desktop app, or `unsloth studio run`).
- Pi 1.4+ (Node 22.19+).

## Usage

| Command | What it does |
|---|---|
| `/unsloth` | Interactive picker: choose a model to load or unload, refresh the list. |
| `/unsloth list` | List every locally available model on the server. |
| `/unsloth status` | Server URL, active model, context, reasoning/vision support. |
| `/unsloth load <model>` | Load a model. Options: `--variant Q8_0`, `--context 65536`. |
| `/unsloth unload <model>` | Unload a model (`--force` skips confirmation). |
| `/unsloth refresh` | Refresh the Pi model catalog. |
| `/login unsloth` | Configure the server URL and an optional API key. |

Model names can be full ids (`unsloth/flint-GGUF`), display names, or ownerless suffixes (`flint-GGUF`).

After a load completes, the model is available in `/model` immediately — switch to it with `/model` or `Ctrl+P`.

## Authentication

Credential resolution, in order:

1. An API key stored through `/login unsloth`.
2. The `UNSLOTH_API_KEY` environment variable.
3. A **desktop token** self-issued through Unsloth's own CLI helpers (`unsloth_cli._inference`: `find_studio_server`, `verify_studio_identity`, `_studio_token`), run from the bundled `unsloth_studio` Python venv. Tokens are minted only for a verified loopback server — never sent to a non-local endpoint — and are cached until shortly before their JWT expiry.

Server discovery honors `UNSLOTH_STUDIO_URL` and defaults to `http://127.0.0.1:8888`. `UNSLOTH_STUDIO_HOME`, `UNSLOTH_HOME`, or `STUDIO_HOME` relocate the Studio home; `UNSLOTH_PYTHON` pins the venv interpreter.

## What's exposed in /model

Only **loaded, chat-capable** models appear as Pi models. Image/audio models are listed by `/unsloth list` but are not chat models. Context windows come from the server's per-model metadata; output caps mirror `unsloth_cli`'s own limits (`min(window/4, 32000)`). Reasoning is enabled only for the active model when the server reports `supports_reasoning`, using the `qwen-chat-template` thinking format the GGUF backend consumes (`chat_template_kwargs.enable_thinking`).

## Development

```bash
npm install
npm test          # node:test against an in-process fake Studio server
npm run typecheck # tsc against the pi-ai / pi-coding-agent types
```

The REST client, catalog mapping, and auth logic are dependency-free modules (`extensions/unsloth/`) so the test suite never loads pi-ai. CI runs the tests and typecheck on Node 22 and 24.

## Repository policy

`main` is protected: no direct pushes, changes land through pull requests with a green CI run.
