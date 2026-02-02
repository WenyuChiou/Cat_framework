#!/usr/bin/env node
/**
 * Claude Code hook → (Codex 0.92.0 OR Gemini CLI) → unified diff → git apply
 *
 * Trigger formats:
 *   codex-edit  path/to/file.md | instruction...
 *   gemini-edit path/to/file.md | instruction...
 */

const fs = require("fs");
const { spawnSync } = require("child_process");

function readStdin() {
  try {
    return fs.readFileSync(0, "utf8");
  } catch {
    return "";
  }
}

function sh(cmd, args, opts = {}) {
  return spawnSync(cmd, args, {
    stdio: "pipe",
    encoding: "utf8",
    ...opts
  });
}

// IMPORTANT: do NOT throw; keep startup quiet.
function exitQuietly() {
  process.exit(0);
}

// Fail with user-visible feedback via Claude Code hook protocol
function fail(msg) {
  console.log(JSON.stringify({
    decision: "block",
    reason: `[Hook Error] ${msg}`
  }));
  process.exit(0);
}

function safeJsonParse(s) {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

/** ---------------- Codex JSONL parsing (codex exec --json) ---------------- **/
function parseJsonLines(text) {
  return text
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(Boolean)
    .map(l => safeJsonParse(l))
    .filter(Boolean);
}

/**
 * Extract text from Codex JSONL.
 * Different builds may use different event shapes; we try common fields.
 */
function extractTextFromCodexJsonl(events) {
  let out = "";
  for (const e of events) {
    if (!e || typeof e !== "object") continue;

    // Common patterns
    if (typeof e.text === "string") out += e.text + "\n";
    if (typeof e.output_text === "string") out += e.output_text + "\n";
    if (typeof e.content === "string") out += e.content + "\n";

    // Some tools nest content
    if (Array.isArray(e.content)) {
      for (const c of e.content) {
        if (typeof c === "string") out += c + "\n";
        if (c && typeof c.text === "string") out += c.text + "\n";
      }
    }
  }
  return out.trim();
}

/** ---------------- Unified diff helpers ---------------- **/
function looksLikeUnifiedDiff(s) {
  return typeof s === "string" && s.trim().startsWith("diff --git");
}

// If model returns extra text, attempt to salvage the diff portion.
function salvageDiff(s) {
  if (!s) return "";
  const idx = s.indexOf("diff --git");
  if (idx >= 0) return s.slice(idx).trim();
  return s.trim();
}

/** ---------------- Engine calls ---------------- **/
function buildEditorPrompt({ engine, target, instruction, fileContent }) {
  // Keep prompt strict: ONLY unified diff, no explanations
  return [
    "You are a code editor.",
    "Return ONLY a unified diff patch.",
    "Do not include explanations, markdown fences, or any extra text.",
    "The patch MUST start with: diff --git",
    "",
    `Target file: ${target}`,
    `Instruction: ${instruction}`,
    "",
    "File content:",
    fileContent
  ].join("\n");
}

function runCodex(prompt) {
  // Codex 0.92.0 confirmed working: codex exec --json "..."
  const r = sh("codex", ["exec", "--json", prompt], { cwd: process.cwd() });
  if (r.status !== 0) {
    fail(`codex exec failed:\n${r.stderr || r.stdout || ""}`);
  }
  const events = parseJsonLines(r.stdout || "");
  const text = extractTextFromCodexJsonl(events);
  return text;
}

function runGemini(prompt) {
  // Gemini CLI typical: gemini -p "<prompt>"
  // If your gemini requires different flags, tell me and I’ll adjust.
  const r = sh("gemini", ["-p", prompt], { cwd: process.cwd() });
  if (r.status !== 0) {
    fail(`gemini failed:\n${r.stderr || r.stdout || ""}`);
  }
  // Gemini usually returns plain text
  return (r.stdout || "").trim();
}

/** ---------------- Apply patch ---------------- **/
function writeAndApplyPatch(patchText) {
  const patchDir = ".claude";
  const patchPath = `${patchDir}/tmp_ai.patch`;

  fs.mkdirSync(patchDir, { recursive: true });
  fs.writeFileSync(patchPath, patchText, "utf8");

  // Check first
  const chk = sh("git", ["apply", "--check", patchPath], { cwd: process.cwd() });
  if (chk.status !== 0) {
    fail(`git apply --check failed:\n${chk.stderr || chk.stdout || ""}`);
  }

  // Apply
  const app = sh("git", ["apply", "--whitespace=nowarn", patchPath], { cwd: process.cwd() });
  if (app.status !== 0) {
    fail(`git apply failed:\n${app.stderr || app.stdout || ""}`);
  }
}

/** ---------------- Main ---------------- **/
function main() {
  const raw = readStdin();
  if (!raw || !raw.trim()) exitQuietly();

  const payload = safeJsonParse(raw);
  if (!payload) exitQuietly();

  const text =
    (payload?.prompt ??
      payload?.input ??
      payload?.text ??
      payload?.message ??
      "").trim();

  if (!text) exitQuietly();

  // Trigger: codex-edit ... | ...
  // Trigger: gemini-edit ... | ...
  let engine = null;
  if (/^codex-edit\s+/i.test(text)) engine = "codex";
  else if (/^gemini-edit\s+/i.test(text)) engine = "gemini";
  else exitQuietly();

  const m = text.match(/^(codex-edit|gemini-edit)\s+(.+?)\s*\|\s*(.+)$/i);
  if (!m) fail("Format: codex-edit <file> | <instruction> (or gemini-edit ...)");

  const target = m[2].trim();
  const instruction = m[3].trim();

  // ---- Optional hardening (recommended) ----
  // 1) Restrict folders:
  // if (!target.startsWith("docs/") && !target.startsWith("examples/")) {
  //   fail("Target not allowed (only docs/ or examples/).");
  // }
  // 2) Restrict extensions:
  // if (!target.toLowerCase().endsWith(".md")) {
  //   fail("Only .md files allowed.");
  // }

  let fileContent = "";
  try {
    fileContent = fs.readFileSync(target, "utf8");
  } catch {
    fail(`Cannot read file: ${target}`);
  }

  const prompt = buildEditorPrompt({ engine, target, instruction, fileContent });

  let patchRaw = "";
  if (engine === "codex") patchRaw = runCodex(prompt);
  else patchRaw = runGemini(prompt);

  const patch = salvageDiff(patchRaw);
  if (!looksLikeUnifiedDiff(patch)) {
    fail("Model output is not a unified diff (must start with diff --git).");
  }

  writeAndApplyPatch(patch);

  // Output JSON so Claude Code knows the hook handled this prompt
  console.log(JSON.stringify({
    decision: "block",
    reason: `[Hook] Applied ${engine} patch to ${target}. Check 'git diff ${target}' to review changes.`
  }));
}

main();
