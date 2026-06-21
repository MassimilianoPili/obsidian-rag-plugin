#!/usr/bin/env node
// CLI per interrogare il RAG di Obsidian senza MCP: parla col server REST locale del plugin
// (server.ts: /health, /search, /v1/chat/completions). Richiede Obsidian aperto col server REST
// abilitato (Impostazioni → "Abilita server REST").
//
// Config (porta + API key), in ordine di precedenza:
//   1) flag:  --port 8765 --key <bearer>
//   2) env:   RAG_PORT, RAG_KEY
//   3) file:  --data <path/to/data.json>  oppure env RAG_DATA  (legge serverPort/serverApiKey)
//
// Uso:
//   node rag.mjs health
//   node rag.mjs search "gestione capitoli" -k 8 [--json]
//   node rag.mjs ask "come funziona la verifica inadempienza?"
import { readFileSync } from "node:fs";

const argv = process.argv.slice(2);
const cmd = argv[0];

function flag(name, def) {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def;
}
function has(name) {
  return argv.includes(name);
}
// primo argomento posizionale dopo il comando che non è una flag né valore di flag
function positional() {
  const out = [];
  for (let i = 1; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("-")) {
      i++; // salta il valore della flag
      continue;
    }
    out.push(a);
  }
  return out.join(" ");
}

function config() {
  let port = flag("--port", process.env.RAG_PORT);
  let key = flag("--key", process.env.RAG_KEY);
  const data = flag("--data", process.env.RAG_DATA);
  if ((!port || !key) && data) {
    try {
      const d = JSON.parse(readFileSync(data, "utf8"));
      port = port || d.serverPort;
      key = key || d.serverApiKey;
    } catch (e) {
      die(`impossibile leggere data.json (${data}): ${e.message}`);
    }
  }
  return { port: Number(port) || 8765, key: key || "" };
}

function die(msg, code = 1) {
  process.stderr.write(`rag: ${msg}\n`);
  process.exit(code);
}

async function call(path, init = {}) {
  const { port, key } = config();
  const headers = { ...(init.headers || {}) };
  if (key) headers["Authorization"] = `Bearer ${key}`;
  const url = `http://127.0.0.1:${port}${path}`;
  let res;
  try {
    res = await fetch(url, { ...init, headers });
  } catch (e) {
    die(`server non raggiungibile su ${url} — Obsidian aperto e server REST abilitato? (${e.message})`);
  }
  const text = await res.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    json = { raw: text };
  }
  if (!res.ok) die(`HTTP ${res.status}: ${json.error || text}`, 2);
  return json;
}

function printResults(j) {
  if (has("--json")) {
    process.stdout.write(JSON.stringify(j, null, 2) + "\n");
    return;
  }
  const hits = j.results || [];
  if (!hits.length) {
    process.stdout.write("(nessun risultato)\n");
    return;
  }
  hits.forEach((h, i) => {
    const loc = h.sourceFile + (h.headerPath ? ` · ${h.headerPath}` : "");
    const body = String(h.content || "")
      .replace(/^\[File:[^\n]*\n/, "")
      .trim()
      .split("\n")
      .slice(0, 4)
      .join("\n");
    process.stdout.write(`\n${i + 1}. ${loc}  (score ${h.score})\n${body}\n`);
  });
}

const VERSION = "0.1.0";

const usage = `rag — CLI del RAG locale di Obsidian (Knowledge Base, senza MCP)  v${VERSION}

USO
  rag <comando> [opzioni]

COMANDI
  search "<query>" [-k N]   Ricerca ibrida (BM25 + vettoriale + grafo dei wikilink). Default k=6.
  ask "<query>"             Estratti citati e formattati (endpoint OpenAI-compatible /v1).
  health                    Stato del server: modello, numero di chunk, ready.
  prompt                    Stampa un mini system-prompt + la lista dei tool, da dare a un LLM.
  tools                     Lista dei tool in JSON (per agenti che vogliono il manifest).
  help | -h | --help        Questo aiuto.
  version | --version       Versione.

OPZIONI
  -k, --k N                 Numero di risultati per 'search' (default 6).
  --json                    Output JSON grezzo (per 'search').
  --port N                  Porta del server REST (default 8765).
  --key <bearer>            API key Bearer (per /search e /v1; non serve per /health).
  --data <path>             Legge porta+key dal data.json del plugin.

CONFIG (precedenza: flag > env > file)
  flag:  --port  --key  --data
  env:   RAG_PORT  RAG_KEY  RAG_DATA
  file:  --data <…/.obsidian/plugins/obsidian-rag/data.json>

ESEMPI
  rag health --port 8765
  rag search "verifica inadempienza" -k 8 --data "$RAG_DATA"
  rag ask "come funziona la VI sopra soglia?" --data "$RAG_DATA"
  rag prompt > rag-tools.md     # da iniettare in un agente LLM

NOTE
  Richiede Obsidian aperto col plugin attivo e "server REST" abilitato (default ON).
  Server loopback (127.0.0.1): /health è senza auth, gli altri richiedono Bearer.`;

// Mini system-prompt + tool list: un LLM lo legge per sapere COME interrogare la KB.
const PROMPT = `# RAG locale — Knowledge Base "Diritti Civili (DC)"

Hai accesso a una Knowledge Base tecnica/funzionale indicizzata con ricerca IBRIDA
(BM25 lessicale + vettoriale semantica + boost dal grafo dei wikilink). Interrogala con:

TOOL
- rag search "<query>" [-k N]   → top-N estratti: "file · sezione" + testo. Default N=6.
- rag ask "<query>"             → estratti citati già formattati (per comporre una risposta).
- rag health                    → stato (modello, n. chunk).

COME INTERROGARE BENE
1. Query in italiano, in linguaggio naturale e SPECIFICHE. I prefissi del modello
   (query:/passage:) sono gestiti internamente: non aggiungerli.
2. Usa i TERMINI DI DOMINIO e i loro alias: le entità sono indicizzate con i sinonimi
   (es. "VI" = verifica inadempienza; "OP" = ordine di pagamento; "ODA" = ordinanza di
   assegnazione). Cercare l'acronimo o il nome esteso funziona ugualmente.
3. Calibra -k: domande ampie/esplorative → -k 8..12; fatti puntuali → -k 3..5.
4. Se la prima query non basta, RIFORMULA con sinonimi o scomponila in sotto-domande
   (multi-query) invece di insistere con le stesse parole.
5. I risultati citano "file · sezione": cita la fonte e, se presenti, rispetta i campi
   di provenance inline ^[source:: …] ^[confidence:: …]. NON inventare: se gli estratti
   non coprono la domanda, dillo o raffina la ricerca.

WORKFLOW CONSIGLIATO
  prima 'search' per individuare le note rilevanti → poi 'ask' (o 'search -k' più alto)
  sulle note/sezioni emerse per estrarre il dettaglio da citare.`;

// Manifest dei tool (JSON) per agenti che preferiscono un descrittore strutturato.
const TOOLS = [
  {
    name: "search",
    description:
      "Ricerca ibrida (BM25+vettoriale+grafo) sulla KB. Ritorna i top-k estratti con file, sezione, testo, score.",
    invoke: 'rag search "<query>" [-k N] [--json]',
    http: "GET /search?q=<query>&k=<N>  (Bearer)",
    params: { query: "stringa, linguaggio naturale", k: "int, default 6" },
  },
  {
    name: "ask",
    description: "Come search ma ritorna estratti citati già formattati (endpoint OpenAI-compatible).",
    invoke: 'rag ask "<query>"',
    http: "POST /v1/chat/completions {messages:[{role:'user',content:'<query>'}]}  (Bearer)",
    params: { query: "stringa" },
  },
  {
    name: "health",
    description: "Stato del server: modello, numero di chunk, ready.",
    invoke: "rag health",
    http: "GET /health  (no auth)",
    params: {},
  },
];

const run = {
  async health() {
    const j = await call("/health");
    process.stdout.write(JSON.stringify(j, null, 2) + "\n");
  },
  async search() {
    const q = positional();
    if (!q) die('manca la query: rag search "..."');
    const k = flag("-k", flag("--k", "6"));
    const j = await call(`/search?q=${encodeURIComponent(q)}&k=${encodeURIComponent(k)}`);
    printResults(j);
  },
  async ask() {
    const q = positional();
    if (!q) die('manca la query: rag ask "..."');
    const j = await call("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: q }] }),
    });
    const content = j.choices?.[0]?.message?.content ?? JSON.stringify(j);
    process.stdout.write(content + "\n");
  },
  prompt() {
    process.stdout.write(PROMPT + "\n");
  },
  tools() {
    process.stdout.write(JSON.stringify(TOOLS, null, 2) + "\n");
  },
  version() {
    process.stdout.write("rag " + VERSION + "\n");
  },
  help() {
    process.stdout.write(usage + "\n");
  },
};

if (cmd === "version" || has("--version")) {
  process.stdout.write("rag " + VERSION + "\n");
  process.exit(0);
}
if (!cmd || cmd === "help" || has("-h") || has("--help") || !run[cmd]) {
  process.stdout.write(usage + "\n");
  process.exit(cmd && !run[cmd] ? 1 : 0);
}
run[cmd]();
