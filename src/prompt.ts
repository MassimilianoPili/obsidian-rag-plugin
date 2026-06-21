// Sorgente unica del mini system-prompt e del manifest dei tool, esposti da Obsidian:
//  - server REST:  GET /prompt (text)  e  GET /tools (json)
//  - comando:      "RAG: copia prompt per agente"
//  - CLI:          rag prompt / rag tools (li recupera dal server)

export const RAG_PROMPT = `# RAG locale — Knowledge Base "Diritti Civili (DC)"

Hai accesso a una Knowledge Base tecnica/funzionale con ricerca IBRIDA (BM25 lessicale +
embedding E5 + boost dal grafo dei wikilink, fusione RRF, MMR per diversità), esposta dal
plugin Obsidian su 127.0.0.1.

TOOL (REST, Bearer <API key> tranne /health,/prompt,/tools · equivalenti CLI tra parentesi)
- GET  /search?q=<query>&k=<N>             → top-N estratti {sourceFile, headerPath, content, score}   (rag search "<q>" -k N)
- POST /v1/chat/completions {messages:[…]} → estratti citati formattati                                 (rag ask "<q>")
- GET  /health                             → {model, chunks, ready}                                      (rag health)

FORMULARE LA QUERY (BM25+denso ⇒ complementari)
- Scrivi FRASI in linguaggio naturale, in italiano, < ~30 parole. NON keyword telegrafiche
  e non paragrafi (diluiscono l'embedding). [E5; MTEB]
- Concetti astratti → domanda naturale (il denso eccelle). Entità precise (nomi classe,
  codici, sigle) → mettile LETTERALMENTE: il BM25 serve per il match esatto. [BEIR; entity-questions]
- Acronimi: includi sigla + forma estesa tra parentesi, es. "VI (verifica inadempienza)".
  Le entità della KB hanno alias indicizzati (VI, OP, ODA…): sigla o nome esteso vanno entrambi. [BM25/F&T]

QUANTI RISULTATI
- k default 6; ampie/esplorative → k 10..12; fatti puntuali → k 3..5.
- Per COMPORRE la risposta usa solo i TOP 3-5 chunk più pertinenti, non tutti: troppo
  contesto degrada la risposta ("lost in the middle"). [Liu 2023]

SE I RISULTATI SONO POVERI (max 3 tentativi, poi astieniti)
- Riformula (keyword↔frase, sinonimi). Multi-query (2-3 riformulazioni, unisci+dedup) se
  <3 risultati o terminologia variabile. [Ma 2023]
- Domanda multi-hop → SCOMPONILA in sotto-domande, cerca ciascuna, poi sintetizza. [self-ask]
- HyDE (documento ipotetico) SOLO per query astratte senza ancore precise; mai su query
  fattuali con entità. [Gao HyDE]

GROUNDING (anti-allucinazione) [ALCE; Self-RAG]
- CITA la fonte per ogni affermazione: "file · sezione" o [[Nome Nota]]. Rispetta i campi
  di provenance inline ^[source:: …] ^[confidence:: …] se presenti.
- NON inventare: se gli estratti non coprono la domanda, dichiaralo. Distingui TROVATO-in-KB
  da INFERENZA da conoscenza GENERALE. Se due chunk si contraddicono, riporta entrambi con fonte.

WORKFLOW: prima /search per trovare le note → poi /ask (o /search con k più alto) sulle
sezioni emerse per estrarre il dettaglio da citare.`;

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
