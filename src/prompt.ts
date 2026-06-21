// Sorgente unica del mini system-prompt e del manifest dei tool, esposti da Obsidian:
//  - server REST:  GET /prompt (text)  e  GET /tools (json)
//  - comando:      "RAG: copia prompt per agente"
//  - CLI:          rag prompt / rag tools (li recupera dal server)

export const RAG_PROMPT = `# RAG locale — Knowledge Base "Diritti Civili (DC)"

Hai accesso a una Knowledge Base tecnica/funzionale indicizzata con ricerca IBRIDA
(BM25 lessicale + vettoriale semantica + boost dal grafo dei wikilink), esposta dal
plugin Obsidian su 127.0.0.1. Strumenti:

TOOL (REST, header Authorization: Bearer <API key> tranne /health)
- GET  /search?q=<query>&k=<N>            → top-N estratti: {sourceFile, headerPath, content, score}
- POST /v1/chat/completions {messages:[…]} → estratti citati formattati (OpenAI-compatible)
- GET  /health                            → {model, chunks, ready}
(Equivalenti CLI: rag search "<q>" -k N | rag ask "<q>" | rag health)

COME INTERROGARE BENE
1. Query in italiano, in linguaggio naturale e SPECIFICHE. I prefissi del modello
   (query:/passage:) sono gestiti internamente: non aggiungerli.
2. Usa i TERMINI DI DOMINIO e i loro alias: le entità sono indicizzate coi sinonimi
   (es. "VI" = verifica inadempienza; "OP" = ordine di pagamento; "ODA" = ordinanza di
   assegnazione). Acronimo o nome esteso funzionano ugualmente.
3. Calibra k: domande ampie/esplorative → k 8..12; fatti puntuali → k 3..5.
4. Se la prima query non basta, RIFORMULA con sinonimi o scomponila in sotto-domande
   (multi-query) invece di insistere con le stesse parole.
5. I risultati citano "file · sezione": cita la fonte e rispetta i campi di provenance
   inline ^[source:: …] ^[confidence:: …] se presenti. NON inventare: se gli estratti
   non coprono la domanda, raffina la ricerca o dichiara l'incertezza.

WORKFLOW CONSIGLIATO
  prima /search per individuare le note rilevanti → poi /v1 (o /search con k più alto)
  sulle note/sezioni emerse per estrarre il dettaglio da citare.`;

export interface RagTool {
  name: string;
  description: string;
  http: string;
  cli: string;
  params: Record<string, string>;
}

export const RAG_TOOLS: RagTool[] = [
  {
    name: "search",
    description:
      "Ricerca ibrida (BM25 + vettoriale + grafo) sulla KB. Ritorna i top-k estratti con file, sezione, testo, score.",
    http: "GET /search?q=<query>&k=<N>",
    cli: 'rag search "<query>" [-k N] [--json]',
    params: { query: "stringa, linguaggio naturale", k: "int, default 6" },
  },
  {
    name: "ask",
    description: "Come search ma ritorna estratti citati già formattati (endpoint OpenAI-compatible).",
    http: "POST /v1/chat/completions {messages:[{role:'user',content:'<query>'}]}",
    cli: 'rag ask "<query>"',
    params: { query: "stringa" },
  },
  {
    name: "health",
    description: "Stato del server: modello, numero di chunk, ready. Nessuna auth.",
    http: "GET /health",
    cli: "rag health",
    params: {},
  },
];
