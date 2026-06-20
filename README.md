# Obsidian RAG (hybrid)

Ricerca **ibrida locale** sulle tue note Obsidian, dentro Obsidian, **senza server né cloud**:

- **BM25** (lessicale, via Orama) ⊕ **semantica** (embedding ONNX locale via transformers.js)
  fusi con **RRF**, + **boost dal grafo** dei link (nativo di Obsidian, `resolvedLinks`).
- Modello embedding multilingua leggero (`Xenova/multilingual-e5-small`, 384-dim) — adatto all'italiano.
- Indicizzazione **incrementale** sugli eventi del vault, indice persistito su file.
- **Server REST locale opt-in** → Claude / CLI / qualunque client OpenAI possono interrogare le note.

Cross-platform desktop (Windows/Mac/Linux): un solo `main.js`, niente binari nativi
(onnxruntime gira in WASM).

## Installazione (sideload)

Copia **3 file** in `<vault>/.obsidian/plugins/obsidian-rag/`:

- `manifest.json`
- `main.js`
- `styles.css`

Poi: Obsidian → Impostazioni → *Community plugins* → attiva **Obsidian RAG (hybrid)**.

> Primo avvio: scarica modello + runtime WASM da CDN (~50-80 MB, serve internet la prima
> volta, poi in cache), quindi indicizza il vault. Una `Notice` segnala "RAG pronto · N chunk".
> Apri il pannello dalla ribbon (icona lente) o col comando *RAG: apri pannello ricerca*.

## Server REST per Claude / CLI (opt-in)

Impostazioni del plugin → *Server REST locale* → **Abilita** (porta default `8765`, API key opzionale).
Con Obsidian aperto:

```bash
curl 'http://127.0.0.1:8765/search?q=come%20faccio%20il%20backup&k=5'
curl http://127.0.0.1:8765/health
# OpenAI-compatible (solo estratti citati):
curl -XPOST http://127.0.0.1:8765/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"obsidian-rag","messages":[{"role":"user","content":"autenticazione SSO"}]}'
```

Con API key: aggiungi `-H "Authorization: Bearer <key>"`. Claude Code può usarlo via `curl`,
come tool REST, o avvolto in un piccolo MCP server. Endpoint legato a `127.0.0.1` (non esposto in rete).

## Impostazioni

| Voce | Default | Note |
|------|---------|------|
| Modello embedding | `Xenova/multilingual-e5-small` | cambiarlo → reindex (dim diversa) |
| Risultati (top-K) | 6 | |
| Boost di grafo | 1.12 | 1 = disattiva il boost dei vicini-di-grafo |
| Server REST | off | porta + Bearer key opzionale |

## Architettura

| File | Ruolo |
|------|-------|
| `chunker.ts` | split heading-aware (code-fence safe) + context prefix `[File][Sezione]` |
| `embedder.ts` | transformers.js (ONNX/WASM), prefissi e5 `query:`/`passage:` |
| `store.ts` | Orama: BM25 + vettori, RRF, graph boost, persistenza save/load |
| `indexer.ts` | walk vault, incrementale per hash, grafo da `resolvedLinks` |
| `view.ts` | pannello ricerca (render markdown degli estratti) |
| `server.ts` | server REST locale (opt-in) |
| `main.ts` | wiring: modello+index in background, eventi vault, comandi, settings |

Origine: il design (chunking, RRF ibrido, graph boost, scelta modello) è stato prima
validato in un servizio Python/Docker (`/data/massimiliano/obsidian-rag/`, ora **blueprint**),
poi riscritto nativo in TS per girare dentro Obsidian.

## Build da sorgente

```bash
npm install
npm run build   # -> main.js
```
