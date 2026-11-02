# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/): tags are immutable and named
`vX.Y.Z`, matching `package.json`.

## 0.2.0

First properly versioned release. Earlier commits in this repository were
development snapshots and were never tagged semver.

### Changed

- **Server selection is deterministic.** The plugin chooses the base URL itself
  (stored credential → `UNSLOTH_STUDIO_URL` → `http://127.0.0.1:8888`) and never
  accepts a base chosen by a helper. `unsloth_cli.find_studio_server()` is no
  longer used, because it prefers "the server this install recorded" and can
  answer with a port nobody asked about when more than one server is running.

### Added

- Three gates before a server is used: `/api/health` reachability, identity
  proof bound to that exact ip+port, and an authenticated request to
  `/api/inference/loaded-models`.
- Explicit failures when a configured server is unreachable or rejects
  credentials. There is no silent fallback to a different port.
- Minted tokens are cached per base URL, so changing `UNSLOTH_STUDIO_URL`
  mid-session cannot reuse another server's token.
- Headless server support documented and verified against
  `unsloth studio --api-only -p 8899`.

### Fixed

- `mintToken()` accepted an `env` option but did not pass it to the spawned
  Python helper, so `UNSLOTH_STUDIO_URL` was ignored by token minting and
  non-default ports silently targeted `:8888`.
- Padded load/unload responses are read to completion, and in-band
  `_deferred_error` reports are surfaced as failures.
