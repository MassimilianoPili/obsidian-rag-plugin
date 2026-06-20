import { Component, ItemView, MarkdownRenderer, TFile, WorkspaceLeaf } from "obsidian";
import type ObsidianRagPlugin from "./main";

export const VIEW_TYPE_RAG = "obsidian-rag-view";

export class RagView extends ItemView {
  constructor(leaf: WorkspaceLeaf, private plugin: ObsidianRagPlugin) {
    super(leaf);
  }

  getViewType() {
    return VIEW_TYPE_RAG;
  }
  getDisplayText() {
    return "RAG ricerca";
  }
  getIcon() {
    return "search";
  }

  async onOpen() {
    const root = this.contentEl;
    root.empty();
    root.addClass("rag-view");

    const header = root.createDiv({ cls: "rag-header" });
    const input = header.createEl("input", { type: "text", cls: "rag-input", placeholder: "Cerca nelle note…" });
    const btn = header.createEl("button", { text: "Cerca", cls: "rag-btn" });
    const status = root.createDiv({ cls: "rag-status", text: this.plugin.modeText() });
    const results = root.createDiv({ cls: "rag-results" });

    const run = async () => {
      const q = input.value.trim();
      if (!q) return;
      results.empty();
      status.setText("cerco…");
      const hits = await this.plugin.search(q);
      if (hits === null) {
        status.setText("Indice non ancora pronto (modello in caricamento).");
        return;
      }
      status.setText(`${hits.length} risultati · ${this.plugin.modeText()}`);
      for (const h of hits) {
        const card = results.createDiv({ cls: "rag-card" });
        const title = card.createDiv({ cls: "rag-card-title" });
        const link = title.createEl("a", { text: h.sourceFile, href: "#" });
        link.onclick = (e) => {
          e.preventDefault();
          this.plugin.openFile(h.sourceFile);
        };
        if (h.headerPath) title.createSpan({ cls: "rag-card-path", text: " · " + h.headerPath });
        const body = card.createDiv({ cls: "rag-card-body" });
        MarkdownRenderer.render(
          this.app,
          this.plugin.stripPrefix(h.content),
          body,
          h.sourceFile,
          this.plugin as unknown as Component,
        );
      }
    };

    btn.onclick = run;
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") run();
    });
    window.setTimeout(() => input.focus(), 0);
  }

  async onClose() {
    this.contentEl.empty();
  }
}
