import { App, Notice, Plugin, PluginSettingTab, Setting, TFile } from "obsidian";
import { Embedder } from "./embedder";
import { Indexer } from "./indexer";
import { ragLog } from "./logger";
import { RagServer } from "./server";
import { HybridStore, SearchResult } from "./store";
import { RagView, VIEW_TYPE_RAG } from "./view";

export interface RagSettings {
  embedModel: string;
  modelConfirmed: boolean; // il modello si scarica/carica solo dopo conferma esplicita dell'utente
  autoLoadOnStartup: boolean; // se true carica modello+indice all'avvio (può rallentare Obsidian)
  topK: number;
  graphBoost: number;
  enableServer: boolean;
  serverPort: number;
  serverApiKey: string;
}

// Modelli suggeriti per la tendina (dim e size indicative). "custom" per HF id arbitrario.
export const SUGGESTED_MODELS: { id: string; label: string }[] = [
  { id: "Xenova/multilingual-e5-small", label: "multilingual-e5-small · 384d · ~120MB · IT, leggero (consigliato)" },
  { id: "Xenova/multilingual-e5-base", label: "multilingual-e5-base · 768d · ~280MB · IT, qualità superiore" },
  { id: "Xenova/paraphrase-multilingual-MiniLM-L12-v2", label: "paraphrase-multilingual-MiniLM-L12 · 384d · IT/multi" },
  { id: "Xenova/bge-small-en-v1.5", label: "bge-small-en-v1.5 · 384d · solo EN" },
  { id: "Xenova/all-MiniLM-L6-v2", label: "all-MiniLM-L6-v2 · 384d · EN, generico" },
];

export const DEFAULT_SETTINGS: RagSettings = {
  embedModel: "Xenova/multilingual-e5-small", // pre-selezionato nella tendina, NON scaricato finché non confermi
  modelConfirmed: false,
  autoLoadOnStartup: false, // default: nessun caricamento all'avvio → apertura Obsidian leggera
  topK: 6,
  graphBoost: 1.12,
  enableServer: false, // opt-in: server REST locale per Claude/CLI
  serverPort: 8765,
  serverApiKey: "",
};

export default class ObsidianRagPlugin extends Plugin {
  settings: RagSettings = DEFAULT_SETTINGS;
  embedder = new Embedder();
  store = new HybridStore();
  indexer!: Indexer;
  server = new RagServer(this);

  async onload() {
    await this.loadSettings();
    this.indexer = new Indexer(this.app, this, this.store, this.embedder);

    this.registerView(VIEW_TYPE_RAG, (leaf) => new RagView(leaf, this));
    this.addRibbonIcon("search", "Obsidian RAG", () => this.activateView());
    this.addCommand({ id: "rag-open", name: "Apri pannello ricerca", callback: () => this.activateView() });
    this.addCommand({ id: "rag-reindex", name: "Reindicizza tutto il vault", callback: () => this.reindex(true) });
    this.addCommand({ id: "rag-test-model", name: "Testa modello embedding", callback: () => this.testModel() });
    this.addCommand({
      id: "rag-copy-log",
      name: "Copia log negli appunti",
      callback: async () => {
        await navigator.clipboard.writeText(ragLog.format());
        new Notice("RAG: log copiato negli appunti.");
      },
    });
    this.addSettingTab(new RagSettingTab(this.app, this));
    ragLog.info("plugin caricato");

    // reindicizzazione incrementale sugli eventi del vault
    this.registerEvent(this.app.vault.on("modify", (f) => f instanceof TFile && this.indexer.reindexFile(f)));
    this.registerEvent(this.app.vault.on("create", (f) => f instanceof TFile && this.indexer.reindexFile(f)));
    this.registerEvent(this.app.vault.on("delete", (f) => this.indexer.removeFile(f.path)));
    this.registerEvent(
      this.app.vault.on("rename", (f, old) => {
        this.indexer.removeFile(old);
        if (f instanceof TFile) this.indexer.reindexFile(f);
      }),
    );

    // modello + indice in background: non blocca l'avvio di Obsidian
    this.app.workspace.onLayoutReady(() => void this.init());
  }

  onunload() {
    this.server.stop();
    if (this.indexer) void this.indexer.flushPersist(); // salva una persist debounced pendente
  }

  private async init() {
    // Niente caricamento all'avvio salvo opt-in esplicito: evita il lag all'apertura di Obsidian.
    // Di default si carica a richiesta dal bottone «Carica modello».
    if (this.settings.modelConfirmed && this.settings.autoLoadOnStartup) {
      await this.loadModelAndIndex();
    }
  }

  /** Scarica/attiva il modello selezionato e (re)indicizza. Invocato dal bottone o all'avvio se già confermato. */
  async loadModelAndIndex() {
    if (this.embedder.loading) {
      new Notice("RAG: caricamento del modello già in corso…");
      return;
    }
    try {
      const short = this.settings.embedModel.split("/").pop();
      new Notice(`RAG: carico «${short}» (download dal CDN al primo uso)…`, 6000);
      await this.embedder.load(this.settings.embedModel, this.onModelProgress);
      const loaded = await this.indexer.tryLoad(this.embedder.model, this.embedder.dim);
      if (!loaded) {
        new Notice("RAG: indicizzo il vault…");
        await this.indexer.reindexAll(false);
      }
      ragLog.info(`pronto · ${this.store.count()} chunk indicizzati`);
      new Notice(`RAG pronto · ${this.store.count()} chunk`);
      if (this.settings.enableServer) this.startServer();
    } catch (e) {
      ragLog.error("loadModelAndIndex", e);
      new Notice("RAG: errore nel caricamento del modello — vedi la sezione «Log» nelle impostazioni.", 8000);
    }
  }

  startServer() {
    try {
      if (!this.settings.serverApiKey) {
        // Genera una API key di default: l'endpoint loopback serve dati privati, niente no-auth.
        const c: any = (globalThis as any).crypto;
        this.settings.serverApiKey = c?.randomUUID
          ? c.randomUUID().replace(/-/g, "")
          : Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2);
        void this.saveSettings();
      }
      this.server.start(this.settings.serverPort, this.settings.serverApiKey);
      new Notice(`RAG: server REST su 127.0.0.1:${this.settings.serverPort} (API key nelle impostazioni)`);
    } catch (e) {
      ragLog.error("startServer", e);
      new Notice("RAG: impossibile avviare il server — vedi «Log» nelle impostazioni.");
    }
  }

  // Progresso download modello → Log (throttled a step del 10% per file, così non spamma).
  private lastPct: Record<string, number> = {};
  onModelProgress = (p: any) => {
    if (!p) return;
    if (p.status === "progress" && p.file) {
      const pct = Math.floor(p.progress ?? 0);
      const bucket = Math.floor(pct / 10) * 10;
      if ((this.lastPct[p.file] ?? -1) < bucket) {
        this.lastPct[p.file] = bucket;
        ragLog.info(`download ${p.file}: ${pct}%`);
      }
    } else if (p.status === "done" && p.file) {
      ragLog.info(`scaricato ${p.file}`);
    }
  };

  /** Verifica rapida del modello selezionato: lo carica se serve ed esegue un embedding di prova (no reindex). */
  async testModel() {
    if (this.embedder.loading) {
      new Notice("RAG: caricamento del modello già in corso…");
      return;
    }
    const short = this.settings.embedModel.split("/").pop();
    try {
      if (!this.embedder.ready || this.embedder.model !== this.settings.embedModel) {
        new Notice(`RAG test: carico «${short}» (download al primo uso)…`, 6000);
        await this.embedder.load(this.settings.embedModel, this.onModelProgress);
      }
      const t0 = performance.now();
      const vec = await this.embedder.embedQuery("prova di funzionamento del modello");
      const ms = Math.round(performance.now() - t0);
      ragLog.info(`test OK · ${short} · dim ${vec.length} · ${ms}ms`);
      new Notice(`RAG test OK · ${short} · dim ${vec.length} · ${ms}ms`, 8000);
    } catch (e) {
      ragLog.error(`testModel «${short}»`, e);
      new Notice(`RAG test: «${short}» NON funzionante — vedi «Log» nelle impostazioni.`, 8000);
    }
  }

  async reindex(force: boolean) {
    if (!this.embedder.ready) {
      new Notice("RAG: modello non ancora pronto.");
      return;
    }
    new Notice("RAG: reindicizzo…");
    await this.indexer.reindexAll(force);
    new Notice(`RAG: ${this.store.count()} chunk indicizzati.`);
  }

  async search(q: string, k = this.settings.topK): Promise<SearchResult[] | null> {
    if (!q.trim() || !this.store.ready || !this.embedder.ready) return null;
    const vec = await this.embedder.embedQuery(q);
    return this.store.search(q, vec, k, this.indexer.neighborsOf, this.settings.graphBoost);
  }

  /** Formatta gli estratti citati come risposta testuale (modalità solo-estratti). */
  formatExtracts(q: string, hits: SearchResult[]): string {
    if (!hits.length) return `Nessun estratto pertinente per «${q}».`;
    const parts = [`Estratti dalle note per: «${q}»\n`];
    hits.forEach((h, i) => {
      const loc = h.sourceFile + (h.headerPath ? ` · ${h.headerPath}` : "");
      parts.push(`**${i + 1}. ${loc}**\n${this.stripPrefix(h.content).trim()}\n`);
    });
    return parts.join("\n");
  }

  modeText(): string {
    return this.embedder.ready ? `ibrido · ${this.embedder.model.split("/").pop()}` : "modello in caricamento…";
  }

  stripPrefix(c: string): string {
    return c.startsWith("[File:") ? c.slice(c.indexOf("\n") + 1) : c;
  }

  openFile(path: string) {
    const f = this.app.vault.getAbstractFileByPath(path);
    if (f instanceof TFile) this.app.workspace.getLeaf(false).openFile(f);
  }

  async activateView() {
    const { workspace } = this.app;
    let leaf = workspace.getLeavesOfType(VIEW_TYPE_RAG)[0];
    if (!leaf) {
      leaf = workspace.getRightLeaf(false)!;
      await leaf.setViewState({ type: VIEW_TYPE_RAG, active: true });
    }
    workspace.revealLeaf(leaf);
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }
  async saveSettings() {
    await this.saveData(this.settings);
  }
}

class RagSettingTab extends PluginSettingTab {
  private logUnsub: (() => void) | null = null;

  constructor(app: App, private plugin: ObsidianRagPlugin) {
    super(app, plugin);
  }

  hide(): void {
    this.logUnsub?.();
    this.logUnsub = null;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h3", { text: "Obsidian RAG — ricerca ibrida locale" });

    const isSuggested = SUGGESTED_MODELS.some((m) => m.id === this.plugin.settings.embedModel);
    let customRow: Setting | null = null;

    new Setting(containerEl)
      .setName("Modello embedding")
      .setDesc("Scegli un modello (multilingua per note italiane). Il download parte SOLO con «Carica modello», non in automatico.")
      .addDropdown((d) => {
        for (const m of SUGGESTED_MODELS) d.addOption(m.id, m.label);
        d.addOption("__custom__", "Custom (HF id)…");
        d.setValue(isSuggested ? this.plugin.settings.embedModel : "__custom__");
        d.onChange(async (v) => {
          if (v === "__custom__") {
            customRow?.settingEl.show();
          } else {
            this.plugin.settings.embedModel = v;
            await this.plugin.saveSettings();
            customRow?.settingEl.hide();
          }
        });
      });

    customRow = new Setting(containerEl)
      .setName("Modello custom")
      .setDesc("Identificatore Hugging Face, es. Xenova/multilingual-e5-base")
      .addText((t) =>
        t.setValue(isSuggested ? "" : this.plugin.settings.embedModel).onChange(async (v) => {
          this.plugin.settings.embedModel = v.trim();
          await this.plugin.saveSettings();
        }),
      );
    if (isSuggested) customRow.settingEl.hide();

    new Setting(containerEl)
      .setName("Carica modello")
      .setDesc("Scarica (se serve) e attiva il modello selezionato, poi indicizza. Dopo aver cambiato modello, premi di nuovo per applicare (reindex).")
      .addButton((b) =>
        b
          .setButtonText(this.plugin.embedder.ready ? "Ricarica / cambia modello" : "Carica modello")
          .setCta()
          .onClick(async () => {
            this.plugin.settings.modelConfirmed = true;
            await this.plugin.saveSettings();
            await this.plugin.loadModelAndIndex();
            this.display(); // refresh stato
          }),
      );

    new Setting(containerEl)
      .setName("Testa modello")
      .setDesc("Carica (se serve) il modello selezionato ed esegue un embedding di prova: verifica che scarichi e funzioni, senza reindicizzare.")
      .addButton((b) =>
        b.setButtonText("Testa").onClick(async () => {
          await this.plugin.testModel();
          this.display(); // refresh stato
        }),
      );

    new Setting(containerEl)
      .setName("Stato modello")
      .setDesc(
        this.plugin.embedder.loading
          ? "Caricamento in corso…"
          : this.plugin.embedder.ready
            ? `Pronto · ${this.plugin.embedder.model} · ${this.plugin.store.count()} chunk`
            : "Non caricato — scegli un modello e premi «Carica modello».",
      );

    new Setting(containerEl)
      .setName("Carica all'avvio di Obsidian")
      .setDesc("Se attivo, all'apertura carica il modello e aggiorna l'indice — può rallentare l'avvio. Default OFF: carica a richiesta col bottone qui sopra.")
      .addToggle((t) =>
        t.setValue(this.plugin.settings.autoLoadOnStartup).onChange(async (v) => {
          this.plugin.settings.autoLoadOnStartup = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Risultati (top-K)")
      .addText((t) =>
        t.setValue(String(this.plugin.settings.topK)).onChange(async (v) => {
          this.plugin.settings.topK = Number(v) || 6;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Boost di grafo")
      .setDesc("Moltiplicatore per i chunk in note collegate ai risultati migliori (1 = disattivato).")
      .addText((t) =>
        t.setValue(String(this.plugin.settings.graphBoost)).onChange(async (v) => {
          this.plugin.settings.graphBoost = Number(v) || 1.12;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Reindicizza ora")
      .addButton((b) => b.setButtonText("Reindex").onClick(() => this.plugin.reindex(true)));

    containerEl.createEl("h3", { text: "Server REST locale (per Claude / CLI)" });
    new Setting(containerEl)
      .setName("Abilita server REST")
      .setDesc("Espone 127.0.0.1 con /search, /health, /v1/chat/completions sullo stesso indice.")
      .addToggle((t) =>
        t.setValue(this.plugin.settings.enableServer).onChange(async (v) => {
          this.plugin.settings.enableServer = v;
          await this.plugin.saveSettings();
          if (v) this.plugin.startServer();
          else this.plugin.server.stop();
        }),
      );
    new Setting(containerEl)
      .setName("Porta")
      .addText((t) =>
        t.setValue(String(this.plugin.settings.serverPort)).onChange(async (v) => {
          this.plugin.settings.serverPort = Number(v) || 8765;
          await this.plugin.saveSettings();
        }),
      );
    new Setting(containerEl)
      .setName("API key (Bearer, opzionale)")
      .setDesc("Se impostata, /search e /v1 richiedono Authorization: Bearer <key>.")
      .addText((t) =>
        t.setValue(this.plugin.settings.serverApiKey).onChange(async (v) => {
          this.plugin.settings.serverApiKey = v.trim();
          await this.plugin.saveSettings();
        }),
      );

    // --- Log / Diagnostica: qui vedi gli errori senza aprire la devtools ---
    containerEl.createEl("h3", { text: "Log / Diagnostica" });
    const logBox = containerEl.createEl("textarea", { cls: "rag-log" });
    logBox.readOnly = true;
    logBox.rows = 14;
    logBox.value = ragLog.format();
    window.setTimeout(() => (logBox.scrollTop = logBox.scrollHeight), 0);
    this.logUnsub?.();
    this.logUnsub = ragLog.subscribe(() => {
      logBox.value = ragLog.format();
      logBox.scrollTop = logBox.scrollHeight;
    });
    new Setting(containerEl)
      .addButton((b) =>
        b.setButtonText("Aggiorna").onClick(() => {
          logBox.value = ragLog.format();
          logBox.scrollTop = logBox.scrollHeight;
        }),
      )
      .addButton((b) =>
        b.setButtonText("Copia").onClick(async () => {
          await navigator.clipboard.writeText(ragLog.format());
          new Notice("Log copiato.");
        }),
      )
      .addButton((b) =>
        b.setButtonText("Pulisci").onClick(() => {
          ragLog.clear();
          logBox.value = "";
        }),
      );
  }
}
