import { App, Notice, Plugin, PluginSettingTab, Setting, TFile } from "obsidian";
import { Embedder } from "./embedder";
import { Indexer } from "./indexer";
import { RagServer } from "./server";
import { HybridStore, SearchResult } from "./store";
import { RagView, VIEW_TYPE_RAG } from "./view";

export interface RagSettings {
  embedModel: string;
  topK: number;
  graphBoost: number;
  enableServer: boolean;
  serverPort: number;
  serverApiKey: string;
}

export const DEFAULT_SETTINGS: RagSettings = {
  embedModel: "Xenova/multilingual-e5-small", // multilingua leggero, adatto a note italiane
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
    this.addSettingTab(new RagSettingTab(this.app, this));

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
  }

  private async init() {
    try {
      new Notice("RAG: carico il modello…");
      await this.embedder.load(this.settings.embedModel);
      const loaded = await this.indexer.tryLoad(this.embedder.model, this.embedder.dim);
      if (!loaded) {
        new Notice("RAG: indicizzo il vault (prima volta)…");
        await this.indexer.reindexAll(false);
      }
      new Notice(`RAG pronto · ${this.store.count()} chunk`);
      if (this.settings.enableServer) this.startServer();
    } catch (e) {
      console.error("RAG init error", e);
      new Notice("RAG: errore in inizializzazione (vedi console).");
    }
  }

  startServer() {
    try {
      this.server.start(this.settings.serverPort, this.settings.serverApiKey);
      new Notice(`RAG: server REST su 127.0.0.1:${this.settings.serverPort}`);
    } catch (e) {
      console.error("RAG server", e);
      new Notice("RAG: impossibile avviare il server (vedi console).");
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
  constructor(app: App, private plugin: ObsidianRagPlugin) {
    super(app, plugin);
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h3", { text: "Obsidian RAG — ricerca ibrida locale" });

    new Setting(containerEl)
      .setName("Modello embedding")
      .setDesc("Modello transformers.js (ONNX). Cambiarlo richiede un reindex (dim diversa).")
      .addText((t) =>
        t.setValue(this.plugin.settings.embedModel).onChange(async (v) => {
          this.plugin.settings.embedModel = v.trim();
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
  }
}
