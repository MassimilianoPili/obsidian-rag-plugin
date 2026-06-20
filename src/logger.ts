// Logger interno del plugin: buffer in memoria + mirror su console + subscribe per UI live.
// Serve a vedere gli errori dentro Obsidian (sezione Log nelle impostazioni e in fondo al pannello),
// senza dover aprire la devtools.

export type LogLevel = "info" | "warn" | "error";

export interface LogEntry {
  t: number;
  level: LogLevel;
  msg: string;
}

type Sub = (e: LogEntry | null) => void;

function partToStr(p: unknown): string {
  if (typeof p === "string") return p;
  if (p instanceof Error) return p.stack || `${p.name}: ${p.message}`;
  try {
    return JSON.stringify(p);
  } catch {
    return String(p);
  }
}

class Logger {
  private buf: LogEntry[] = [];
  private subs = new Set<Sub>();
  private readonly max = 400;

  private push(level: LogLevel, parts: unknown[]) {
    const msg = parts.map(partToStr).join(" ");
    const e: LogEntry = { t: Date.now(), level, msg };
    this.buf.push(e);
    if (this.buf.length > this.max) this.buf.shift();
    const c = level === "error" ? console.error : level === "warn" ? console.warn : console.log;
    c("[RAG]", msg);
    for (const s of this.subs) {
      try {
        s(e);
      } catch {
        /* ignore subscriber errors */
      }
    }
  }

  info(...parts: unknown[]) {
    this.push("info", parts);
  }
  warn(...parts: unknown[]) {
    this.push("warn", parts);
  }
  error(...parts: unknown[]) {
    this.push("error", parts);
  }

  all(): LogEntry[] {
    return [...this.buf];
  }

  clear() {
    this.buf = [];
    for (const s of this.subs) {
      try {
        s(null);
      } catch {
        /* ignore */
      }
    }
  }

  /** Ritorna l'unsubscribe. */
  subscribe(s: Sub): () => void {
    this.subs.add(s);
    return () => this.subs.delete(s);
  }

  format(): string {
    return this.buf
      .map((e) => `${new Date(e.t).toLocaleTimeString()} [${e.level.toUpperCase()}] ${e.msg}`)
      .join("\n");
  }
}

// Singleton condiviso da plugin, view, settings, server.
export const ragLog = new Logger();
