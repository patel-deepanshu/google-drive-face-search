/* Turns whatever the user pastes or drops into clean Google credentials.
 * Pure functions, no DOM: the page (app.js) uses them, and they can be tested on their own. */
(function (root) {
  "use strict";

  function stripJunk(text) {
    return String(text == null ? "" : text)
      .replace(/^﻿/, "")
      .replace(/[“”„‟]/g, '"') // curly quotes from Word / chat apps
      .replace(/[‘’]/g, "'")
      .trim();
  }

  // Copying a key often leaves real line breaks inside a JSON string, which JSON.parse rejects.
  function escapeNewlinesInStrings(text) {
    let out = "", inString = false, escaped = false;
    for (const ch of text) {
      if (!inString) {
        if (ch === '"') inString = true;
        out += ch;
      } else if (escaped) {
        out += ch;
        escaped = false;
      } else if (ch === "\\") {
        out += ch;
        escaped = true;
      } else if (ch === '"') {
        out += ch;
        inString = false;
      } else if (ch === "\n") out += "\\n";
      else if (ch === "\t") out += "\\t";
      else if (ch !== "\r") out += ch;
    }
    return out;
  }

  const dropTrailingCommas = (t) => t.replace(/,\s*([}\]])/g, "$1");

  /** JSON.parse that forgives the usual copy-paste damage. Throws if nothing works. */
  function parseLooseJson(text) {
    let t = stripJunk(text);
    if (!t) throw new Error("Nothing to read.");
    // `"type": "service_account", ...` pasted without the outer braces
    if (!/^[{[]/.test(t) && /^"?[A-Za-z_]+"?\s*:/.test(t)) t = "{" + t + "}";
    const fixed = escapeNewlinesInStrings(t);
    let firstError = null;
    for (const attempt of [t, dropTrailingCommas(t), fixed, dropTrailingCommas(fixed)]) {
      try {
        const value = JSON.parse(attempt);
        // JSON that was itself pasted as a quoted string, e.g. "{\"type\": ...}"
        return typeof value === "string" ? parseLooseJson(value) : value;
      } catch (e) {
        firstError = firstError || e;
      }
    }
    throw firstError;
  }

  /** Make a private key valid PEM: real newlines, 64-character lines, nothing extra around it. */
  function normalizePrivateKey(raw) {
    let key = stripJunk(raw)
      .replace(/^["']+|["']+$/g, "")
      .replace(/\\r/g, "")
      .replace(/\\n/g, "\n") // JSON-escaped newlines
      .replace(/\r/g, "")
      .trim();
    const m = key.match(/-----BEGIN ([A-Z ]*PRIVATE KEY)-----([\s\S]*?)-----END \1-----/);
    if (!m) return key;
    const body = m[2].replace(/\s+/g, "");
    return `-----BEGIN ${m[1]}-----\n${(body.match(/.{1,64}/g) || []).join("\n")}\n-----END ${m[1]}-----\n`;
  }

  function isValidPrivateKey(key) {
    const m = String(key || "").match(/^-----BEGIN [A-Z ]*PRIVATE KEY-----\n([A-Za-z0-9+/=\n]+)\n-----END [A-Z ]*PRIVATE KEY-----\n?$/);
    return !!m && m[1].replace(/\n/g, "").length >= 200;
  }

  const FIELD_NAMES = ["client_email", "private_key", "project_id", "private_key_id", "token_uri"];

  /**
   * Work out what the user gave us.
   * Returns one of:
   *   {kind: "service_account", fields, notes}   fields = client_email, private_key, project_id, ...
   *   {kind: "api_key", apiKey}
   *   {kind: "private_key", privateKey}
   *   {kind: "error", message}
   */
  function parseCredentials(text) {
    const t = stripJunk(text).replace(/^["']+|["']+$/g, "");
    if (!t) return { kind: "error", message: "Nothing to read yet." };

    if (/^AIza[0-9A-Za-z_-]{30,}$/.test(t)) return { kind: "api_key", apiKey: t };
    if (/^-----BEGIN [A-Z ]*PRIVATE KEY-----/.test(t)) { // just the key, nothing else
      return { kind: "private_key", privateKey: normalizePrivateKey(t) };
    }

    let data;
    try {
      data = parseLooseJson(text);
    } catch (e) {
      return { kind: "error", message: "That doesn't look like JSON I can read (" + e.message + "). " +
        "Paste the whole contents of the .json key file, or drop the file here." };
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      return { kind: "error", message: "Expected a JSON object like { \"type\": \"service_account\", … }." };
    }
    if (data.installed || data.web) {
      return { kind: "error", message: "This is an OAuth client file (credentials.json). It can't read a Drive folder by itself. " +
        "Create a Service account key instead (Google Cloud → IAM & Admin → Service Accounts → Keys → Add key → JSON), " +
        "or use an API key for a public folder." };
    }
    if (data.type === "authorized_user" || (data.refresh_token && data.client_id)) {
      return { kind: "error", message: "This is a personal login token, which isn't supported here. " +
        "Use a Service account key (JSON), or an API key for a public folder." };
    }
    if (!data.client_email && !data.private_key) {
      return { kind: "error", message: "I couldn't find client_email or private_key in that JSON. " +
        "Is it a Google service account key file?" };
    }

    const fields = {};
    for (const name of FIELD_NAMES) fields[name] = typeof data[name] === "string" ? data[name].trim() : "";
    fields.private_key = normalizePrivateKey(fields.private_key);
    if (!fields.token_uri) fields.token_uri = "https://oauth2.googleapis.com/token";
    const notes = [];
    if (data.type && data.type !== "service_account") notes.push(`Unexpected type "${data.type}"; treating it as a service account.`);
    return { kind: "service_account", fields, notes };
  }

  /** Problems that would make Google reject the credentials (empty list = fine). */
  function validateServiceAccount(f) {
    const problems = [];
    if (!f.client_email) problems.push("Client email is missing.");
    else if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(f.client_email)) problems.push("Client email doesn't look like an email address.");
    if (!f.private_key) problems.push("Private key is missing.");
    else if (!isValidPrivateKey(normalizePrivateKey(f.private_key))) {
      problems.push("Private key looks incomplete: it should start with -----BEGIN PRIVATE KEY----- and end with -----END PRIVATE KEY-----.");
    }
    return problems;
  }

  /** The exact object the server receives. */
  function buildServiceAccount(f) {
    const out = { type: "service_account", client_email: String(f.client_email || "").trim(),
                  private_key: normalizePrivateKey(f.private_key),
                  token_uri: String(f.token_uri || "").trim() || "https://oauth2.googleapis.com/token" };
    if (f.project_id && f.project_id.trim()) out.project_id = f.project_id.trim();
    if (f.private_key_id && f.private_key_id.trim()) out.private_key_id = f.private_key_id.trim();
    return out;
  }

  /** Folder ID from any common Drive link shape, or a bare ID. Returns null if none found. */
  function extractFolderId(link) {
    const t = String(link || "").trim();
    if (/^[A-Za-z0-9_-]{10,}$/.test(t)) return t;
    const m = t.match(/\/folders\/([A-Za-z0-9_-]{10,})/) || t.match(/[?&]id=([A-Za-z0-9_-]{10,})/);
    return m ? m[1] : null;
  }

  const api = { parseCredentials, parseLooseJson, normalizePrivateKey, isValidPrivateKey,
                validateServiceAccount, buildServiceAccount, extractFolderId };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Creds = api;
})(typeof window !== "undefined" ? window : this);
