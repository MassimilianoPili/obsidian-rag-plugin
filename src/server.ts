// Server REST locale (127.0.0.1) dentro il plugin: stesso indice della UI, interrogabile
// da Claude/CLI o da un client OpenAI-compatible. Opt-in dalle impostazioni.
//
// Sicurezza (server loopback con dati PRIVATI): NIENTE CORS (i chiamanti nativi non ne hanno
// bisogno; abilitarlo permetterebbe a una pagina web di leggere le note); validazione dell'Host
// header (anti DNS-rebinding); body cap + timeout; API key Bearer richiesta di default.
import type ObsidianRagPlugin from "./main";
import { RAG_PROMPT, RAG_TOOLS } from "./prompt";

const BODY_CAP = 1_000_000; // 1 MB
const REQ_TIMEOUT_MS = 15_000;

function readBody(req: any): Promise<any> {
  return new Promise((resolve) => {
    let d = "";
    let total = 0;
    req.on("data", (c: any) => {
      total += c.length;
      if (total > BODY_CAP) {
        try {
          req.destroy();
        } catch {
          /* ignore */
        }
        return resolve({});
      }
      d += c;
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(d || "{}"));
      } catch {
        resolve({});
      }
    });
  });
}

export class RagServer {
  private server: any = null;
  private port = 0;

  constructor(private plugin: ObsidianRagPlugin) {}

  get running() {
    return this.server !== null;
  }
  get boundPort() {
    return this.port;
  }

  start(port: number, apiKey: string) {
    this.stop();
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const http = require("http");
    this.port = port;
    const hostOk = new RegExp(`^(127\\.0\\.0\\.1|localhost|\\[::1\\]):${port}$`);

    this.server = http.createServer(async (req: any, res: any) => {
      req.setTimeout(REQ_TIMEOUT_MS);
      const send = (code: number, obj: unknown) => {
        // Nessun header CORS: il SOP del browser blocca la lettura cross-origin (voluto).
        res.writeHead(code, { "Content-Type": "application/json" });
        res.end(JSON.stringify(obj));
      };
      try {
        // Anti DNS-rebinding: accetta solo Host loopback con la porta attesa.
        if (!hostOk.test((req.headers["host"] || "").toString())) {
          return send(403, { error: "forbidden host" });
        }

        const url = new URL(req.url, "http://localhost");

        // Auth Bearer richiesta su tutto tranne gli endpoint di discovery (no dati sensibili).
        const noAuth = url.pathname === "/health" || url.pathname === "/prompt" || url.pathname === "/tools";
        if (apiKey && !noAuth) {
          if ((req.headers["authorization"] || "") !== `Bearer ${apiKey}`) {
            return send(401, { error: "unauthorized" });
          }
        }

        if (url.pathname === "/health") {
          return send(200, {
            status: "ok",
            model: this.plugin.embedder.model,
            ready: this.plugin.embedder.ready,
            chunks: this.plugin.store.count(),
          });
        }

        if (url.pathname === "/tools") {
          return send(200, { tools: RAG_TOOLS });
        }

        if (url.pathname === "/prompt") {
          // testo semplice: pronto da iniettare in un system prompt
          res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
          res.end(RAG_PROMPT);
          return;
        }

        if (url.pathname === "/search" && req.method === "GET") {
          const q = url.searchParams.get("q") || "";
          const k = Number(url.searchParams.get("k")) || this.plugin.settings.topK;
          const hits = await this.plugin.search(q, k);
          return send(200, { query: q, ready: this.plugin.embedder.ready, count: hits?.length ?? 0, results: hits ?? [] });
        }

        if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
          if (Number(req.headers["content-length"]) > BODY_CAP) return send(413, { error: "payload too large" });
          const body = await readBody(req);
          const msgs: any[] = body.messages || [];
          const lastUser = [...msgs].reverse().find((m) => m.role === "user");
          const q = (lastUser?.content || msgs[msgs.length - 1]?.content || "").toString();
          const hits = (await this.plugin.search(q, 6)) || [];
          const created = Math.floor(Date.now() / 1000);
          return send(200, {
            id: `chatcmpl-${created}`,
            object: "chat.completion",
            created,
            model: body.model || "obsidian-rag",
            choices: [
              { index: 0, message: { role: "assistant", content: this.plugin.formatExtracts(q, hits) }, finish_reason: "stop" },
            ],
          });
        }

        send(404, { error: "not found" });
      } catch (e: any) {
        send(500, { error: String(e?.message || e) });
      }
    });

    this.server.on("error", (e: any) => {
      console.error("RAG server error", e);
      this.server = null;
    });
    this.server.listen(port, "127.0.0.1");
  }

  stop() {
    if (this.server) {
      try {
        this.server.close();
      } catch {
        /* ignore */
      }
      this.server = null;
    }
  }
}
