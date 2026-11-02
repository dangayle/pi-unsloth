// SPDX-License-Identifier: MIT
// Tiny in-process HTTP server for exercising UnslothClient without a real
// Unsloth Studio backend.

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

export interface RecordedRequest {
  method: string;
  url: string;
  headers: Record<string, string | string[] | undefined>;
  body: string;
}

export type FakeHandler = (
  req: IncomingMessage,
  res: ServerResponse,
  recorded: RecordedRequest,
) => void;

export interface FakeServer {
  baseUrl: string;
  requests: RecordedRequest[];
  close(): Promise<void>;
}

export async function startFakeServer(handler: FakeHandler): Promise<FakeServer> {
  const requests: RecordedRequest[] = [];
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk: Buffer | string) => {
      body += chunk.toString();
    });
    req.on("end", () => {
      const recorded: RecordedRequest = {
        method: req.method ?? "GET",
        url: req.url ?? "/",
        headers: req.headers,
        body,
      };
      requests.push(recorded);
      handler(req, res, recorded);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    requests,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

export function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(typeof payload === "string" ? payload : JSON.stringify(payload));
}

export const MODEL_LIST_PAYLOAD = {
  object: "list",
  data: [
    {
      id: "unsloth/Qwen3.5-9B-GGUF",
      object: "model",
      created: 1793597183,
      owned_by: "unsloth-studio",
      quant: "Q8_0",
      context_length: 262144,
      max_context_length: 262144,
      loaded: true,
      status: { value: "loaded" },
    },
    {
      id: "unsloth/flint-GGUF",
      object: "model",
      created: 1793597183,
      owned_by: "unsloth-studio",
      quant: "UD-Q3_K_XL",
      display_name: "flint-GGUF",
      loaded: false,
      status: { value: "unloaded" },
    },
    {
      id: "unsloth/FLUX.2-klein-4B-GGUF",
      object: "model",
      task: "text-to-image",
      quant: "Q4_K_M",
      loaded: true,
      status: { value: "loaded" },
    },
  ],
};

export const STATUS_PAYLOAD = {
  model_identifier: "unsloth/Qwen3.5-9B-GGUF",
  active_model: "unsloth/Qwen3.5-9B-GGUF",
  is_gguf: true,
  gguf_variant: "Q8_0",
  context_length: 262144,
  max_context_length: 262144,
  supports_reasoning: true,
  reasoning_style: "enable_thinking",
  supports_tools: true,
  is_vision: true,
};
