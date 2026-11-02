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

- Unsloth Studio installed and running (the desktop app, or a headless server — see below).
- Pi 1.4+ (Node 22.19+).

## Desktop app, headless, or Core?

The plugin talks to the Unsloth **backend server**, so any mode that serves its REST API works — the desktop app and a headless CLI install are interchangeable:

| Unsloth mode | Works | Notes |
|---|---|---|
| Desktop app | ✅ | Default target, `http://127.0.0.1:8888`. |
| `unsloth studio` / `unsloth studio -p <port>` | ✅ | Same backend, with or without the web UI. |
| `unsloth studio --api-only` | ✅ | No UI at all; verified end-to-end (see below). |
| Docker / remote | ✅ | Point `UNSLOTH_STUDIO_URL` at it and pass an API key (`unsloth studio run` prints one). |
| Unsloth Core (pip library) / in-process CLI inference | ❌ | No HTTP server to attach to. Use Pi's built-in llama.cpp or Ollama support instead. |

### Headless / API-only servers

```bash
unsloth studio --api-only -p 8899
UNSLOTH_STUDIO_URL=http://127.0.0.1:8899 pi
```

`UNSLOTH_STUDIO_URL` is honored exactly. Verified against a headless `unsloth studio --api-only -p 8899` instance driven by this plugin: token minting, model listing, load (53s), catalog reporting the model as loaded, a chat completion through `/v1/chat/completions` (with `reasoning_content`), and unload.

## How the server is chosen

Which server to talk to is decided deterministically, never by a helper:

1. The URL stored with `/login unsloth`.
2. `UNSLOTH_STUDIO_URL`.
3. `http://127.0.0.1:8888`, Unsloth's default.

That order is the whole story — the plugin never lets Unsloth's own discovery pick a port. (`unsloth_cli.find_studio_server()` prefers “the server this install recorded”, so with two servers running it can answer with a port nobody asked about. It is deliberately unused.)

Before a session uses the server, three things must line up:

| Check | Catches |
|---|---|
| `GET /api/health` | A dead port, or a non-Unsloth process squatting on it. |
| Identity proof at that exact address (Unsloth's own `verify_studio_identity`) | A different Unsloth install whose `studio.db` secret does not match. |
| Authenticated `GET /api/inference/loaded-models` | A token the server actually rejects. |

Minted tokens are cached **per base URL**, so changing `UNSLOTH_STUDIO_URL` mid-session forces a fresh mint rather than reusing a token from another server.

When nothing is listening, the failure is explicit — the plugin never quietly falls back to another port:

```
$ UNSLOTH_STUDIO_URL=http://127.0.0.1:9999 pi
Could not authenticate to http://127.0.0.1:9999: the server at that address did not
prove the shared secret from studio.db. Start Unsloth Studio there, point
UNSLOTH_STUDIO_URL at the right port, or set UNSLOTH_API_KEY.
```

A self-issued desktop token is only ever minted for a verified loopback address, so an off-machine server needs `UNSLOTH_API_KEY` (`unsloth studio run` prints one).

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
3. A **desktop token** self-issued through Unsloth's own CLI helpers (`unsloth_cli._inference`: `verify_studio_identity` and `_studio_token`), run from the bundled `unsloth_studio` Python venv. Tokens are minted only for a verified loopback server — never sent to a non-local endpoint — and are cached per base URL until shortly before their JWT expiry.

Server selection honors `UNSLOTH_STUDIO_URL` and defaults to `http://127.0.0.1:8888`; see [How the server is chosen](#how-the-server-is-chosen). `UNSLOTH_STUDIO_HOME`, `UNSLOTH_HOME`, or `STUDIO_HOME` relocate the Studio home; `UNSLOTH_PYTHON` pins the venv interpreter.

## What's exposed in /model

Only **loaded, chat-capable** models appear as Pi models. Image/audio models are listed by `/unsloth list` but are not chat models. Context windows come from the server's per-model metadata; output caps mirror `unsloth_cli`'s own limits (`min(window/4, 32000)`). Reasoning is enabled only for the active model when the server reports `supports_reasoning`, using the `qwen-chat-template` thinking format the GGUF backend consumes (`chat_template_kwargs.enable_thinking`).

## Development

```bash
npm install
npm test          # node:test against an in-process fake Studio server
npm run typecheck # tsc against the pi-ai / pi-coding-agent types
```

The REST client, catalog mapping, and auth logic are dependency-free modules (`extensions/unsloth/`) so the test suite never loads pi-ai. CI runs the tests and typecheck on Node 22 and 24.
