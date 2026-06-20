// Server REST locale (127.0.0.1) dentro il plugin: stesso indice della UI, interrogabile
// da Claude/CLI o da qualunque client OpenAI-compatible. Opt-in dalle impostazioni.
import type ObsidianRagPlugin from "./main";

function readBody(req: any): Promise<any> {
  return new Promise((resolve) => {
    let d = "";
    req.on("data", (c: any) => (d += c));
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

    this.server = http.createServer(async (req: any, res: any) => {
      const cors = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "authorization,content-type",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      };
      const send = (code: number, obj: unknown) => {
        res.writeHead(code, { "Content-Type": "application/json", ...cors });
        res.end(JSON.stringify(obj));
      };
      try {
        if (req.method === "OPTIONS") {
          res.writeHead(204, cors);
          return res.end();
        }
        const url = new URL(req.url, "http://localhost");

        if (apiKey && url.pathname !== "/health") {
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

        if (url.pathname === "/search" && req.method === "GET") {
          const q = url.searchParams.get("q") || "";
          const k = Number(url.searchParams.get("k")) || this.plugin.settings.topK;
          const hits = await this.plugin.search(q, k);
          return send(200, { query: q, ready: this.plugin.embedder.ready, count: hits?.length ?? 0, results: hits ?? [] });
        }

        if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
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
