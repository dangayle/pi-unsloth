// SPDX-License-Identifier: MIT
import { test } from "node:test";
import assert from "node:assert/strict";
import type { UnslothModelEntry, UnslothStatus } from "../extensions/unsloth/client.ts";
import {
  contextWindowOf,
  isChatEntry,
  isLoadedEntry,
  nameOf,
  outputLimit,
  statusMatchesEntry,
  toPiCatalog,
  toPiModel,
} from "../extensions/unsloth/catalog.ts";

const loadedChat: UnslothModelEntry = {
  id: "unsloth/Qwen3.5-9B-GGUF",
  loaded: true,
  status: { value: "loaded" },
  quant: "Q8_0",
  contextLength: 262144,
};

const unloadedChat: UnslothModelEntry = {
  id: "unsloth/flint-GGUF",
  loaded: false,
  status: { value: "unloaded" },
  quant: "UD-Q3_K_XL",
  displayName: "flint-GGUF",
};

const loadedImage: UnslothModelEntry = {
  id: "unsloth/FLUX.2-klein-4B-GGUF",
  loaded: true,
  status: { value: "loaded" },
  task: "text-to-image",
};

const activeStatus: UnslothStatus = {
  modelIdentifier: "unsloth/Qwen3.5-9B-GGUF",
  contextLength: 262144,
  supportsReasoning: true,
  isVision: true,
};

test("isLoadedEntry honors the loaded flag and status value", () => {
  assert.equal(isLoadedEntry(loadedChat), true);
  assert.equal(isLoadedEntry(unloadedChat), false);
  assert.equal(isLoadedEntry({ id: "x", loaded: false, status: { value: "loaded" } }), true);
});

test("isChatEntry excludes known media tasks and admits chat models", () => {
  assert.equal(isChatEntry(loadedChat), true);
  assert.equal(isChatEntry(loadedImage), false);
  assert.equal(isChatEntry({ id: "x", loaded: true, status: {}, task: "text-to-speech" }), false);
  assert.equal(isChatEntry({ id: "x", loaded: true, status: {}, task: "some-future-task" }), true);
});

test("contextWindowOf falls back to 128000", () => {
  assert.equal(contextWindowOf(loadedChat), 262144);
  assert.equal(contextWindowOf({ id: "x", loaded: false, status: {} }), 128000);
  assert.equal(contextWindowOf({ id: "x", loaded: false, status: {}, maxContextLength: 8192 }), 8192);
});

test("outputLimit mirrors unsloth_cli.opencode_output_limit for Pi", () => {
  assert.equal(outputLimit(262144), 32000); // min(window/4, 32000)
  assert.equal(outputLimit(8192), 2048);
  assert.equal(outputLimit(100), 25);
  assert.equal(outputLimit(2), 1);
});

test("nameOf prefers a distinct display name", () => {
  assert.equal(nameOf(loadedChat), "unsloth/Qwen3.5-9B-GGUF");
  assert.equal(nameOf(unloadedChat), "flint-GGUF");
});

test("statusMatchesEntry matches the active model identifier", () => {
  assert.equal(statusMatchesEntry(activeStatus, loadedChat), true);
  assert.equal(statusMatchesEntry(activeStatus, unloadedChat), false);
  assert.equal(statusMatchesEntry({ modelIdentifier: "x" }, { id: "a/x", loaded: true, status: {} }), true);
});

test("toPiModel carries context, zero cost and conservative defaults", () => {
  const model = toPiModel(loadedChat, activeStatus, "http://127.0.0.1:8888/v1");
  assert.equal(model.id, "unsloth/Qwen3.5-9B-GGUF");
  assert.equal(model.api, "openai-completions");
  assert.equal(model.provider, "unsloth");
  assert.equal(model.baseUrl, "http://127.0.0.1:8888/v1");
  assert.equal(model.contextWindow, 262144);
  assert.equal(model.maxTokens, 32000);
  assert.deepEqual(model.cost, { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
  assert.equal(model.reasoning, true);
  assert.deepEqual(model.input, ["text", "image"]);
  assert.equal(model.compat.thinkingFormat, "qwen-chat-template");
  assert.equal(model.compat.maxTokensField, "max_tokens");
  assert.equal(model.compat.supportsStore, false);
});

test("toPiModel is conservative when the status describes another model", () => {
  const model = toPiModel(loadedChat, { modelIdentifier: "unsloth/flint-GGUF", supportsReasoning: true, isVision: true }, "http://127.0.0.1:8888/v1");
  assert.equal(model.reasoning, false);
  assert.deepEqual(model.input, ["text"]);
  assert.equal(model.compat.thinkingFormat, undefined);
});

test("toPiCatalog lists only loaded chat models", () => {
  const catalog = toPiCatalog([loadedChat, unloadedChat, loadedImage], activeStatus, "http://127.0.0.1:8888/v1");
  assert.deepEqual(catalog.map((model) => model.id), ["unsloth/Qwen3.5-9B-GGUF"]);
});
