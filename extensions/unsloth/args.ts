// SPDX-License-Identifier: MIT
// Pure argument parsing and model matching for the /unsloth command.
// Kept free of pi-ai imports so it can be unit-tested standalone.

import type { ModelListEntry } from "./client.ts";

/** Tokenize a command line, honoring single and double quotes. */
export function tokenize(input: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let quote: "'" | '"' | undefined;
  for (let index = 0; index < input.length; index++) {
    const char = input[index];
    if (quote) {
      if (char === quote) {
        quote = undefined;
      } else {
        current += char;
      }
    } else if (char === "'" || char === '"') {
      quote = char;
    } else if (/\s/.test(char)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
    } else {
      current += char;
    }
  }
  if (current) tokens.push(current);
  return tokens;
}

export function flagValue(tokens: readonly string[], flag: string): string | undefined {
  const index = tokens.indexOf(flag);
  return index >= 0 ? tokens[index + 1] : undefined;
}

export function numberFlag(tokens: readonly string[], flag: string): number | undefined {
  const value = flagValue(tokens, flag);
  if (value === undefined) return undefined;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

/** Best-effort match of a user-supplied name against the server's catalog. */
export function findEntry(
  entries: readonly ModelListEntry[],
  name: string,
): ModelListEntry | undefined {
  const wanted = name.toLowerCase();
  return (
    entries.find((entry) => entry.id.toLowerCase() === wanted) ??
    entries.find((entry) => entry.displayName?.toLowerCase() === wanted) ??
    entries.find((entry) => entry.id.toLowerCase().endsWith(`/${wanted}`)) ??
    entries.find((entry) => entry.id.toLowerCase().includes(wanted))
  );
}
