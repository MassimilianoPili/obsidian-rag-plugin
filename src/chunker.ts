// Chunking Markdown — port 1:1 da chunker.py (validato).
// frontmatter-strip -> split per heading (code-fence aware) -> split ricorsivo -> context prefix.

export interface Chunk {
  content: string;
  heading: string;
  headerPath: string;
  chunkIndex: number;
}

const CHARS_PER_TOKEN = 3.5;
const TARGET_CHARS = Math.floor(400 * CHARS_PER_TOKEN); // 1400
const MAX_CHARS = Math.floor(480 * CHARS_PER_TOKEN); // 1680
const SEPARATORS = ["\n\n", "\n", ". ", " ", ""];

const FRONTMATTER_RE = /^---\n[\s\S]*?\n---\n/;
const HEADING_RE = /^(#{1,6})\s+(.*)$/;
const FENCE_RE = /^\s*(```|~~~)/;

export function parseFrontmatter(text: string): { meta: Record<string, string>; body: string } {
  const meta: Record<string, string> = {};
  const m = text.match(FRONTMATTER_RE);
  if (!m) return { meta, body: text };
  for (const line of m[0].split("\n")) {
    if (line.startsWith("---") || !line.includes(":")) continue;
    const i = line.indexOf(":");
    meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return { meta, body: text.slice(m[0].length) };
}

interface Section {
  headerPath: string;
  heading: string;
  text: string;
}

export function splitSections(body: string): Section[] {
  const sections: Section[] = [];
  const stack: Array<[number, string]> = [];
  let curHeading = "";
  let curLines: string[] = [];
  let inFence = false;

  const flush = () => {
    const text = curLines.join("\n").trim();
    if (text) {
      sections.push({ headerPath: stack.map(([, t]) => t).join(" > "), heading: curHeading, text });
    }
  };

  for (const line of body.split("\n")) {
    if (FENCE_RE.test(line)) {
      inFence = !inFence;
      curLines.push(line);
      continue;
    }
    const m = inFence ? null : line.match(HEADING_RE);
    if (m) {
      flush();
      curLines = [];
      const level = m[1].length;
      const title = m[2].trim();
      while (stack.length && stack[stack.length - 1][0] >= level) stack.pop();
      stack.push([level, title]);
      curHeading = title;
    } else {
      curLines.push(line);
    }
  }
  flush();
  return sections;
}

function recursiveSplit(text: string, maxChars: number, sepIndex = 0): string[] {
  if (text.length <= maxChars) return [text];
  const sep = sepIndex < SEPARATORS.length ? SEPARATORS[sepIndex] : "";
  if (sep === "") {
    const out: string[] = [];
    for (let i = 0; i < text.length; i += maxChars) out.push(text.slice(i, i + maxChars));
    return out;
  }
  const chunks: string[] = [];
  let buf = "";
  for (const part of text.split(sep)) {
    const piece = part + sep;
    if (buf.length + piece.length <= maxChars) {
      buf += piece;
    } else {
      if (buf) chunks.push(buf);
      if (piece.length > maxChars) {
        chunks.push(...recursiveSplit(part, maxChars, sepIndex + 1));
        buf = "";
      } else {
        buf = piece;
      }
    }
  }
  if (buf.trim()) chunks.push(buf);
  return chunks.map((c) => c.trim()).filter((c) => c.length > 0);
}

function mergeSmall(chunks: string[], target: number): string[] {
  const merged: string[] = [];
  let buf = "";
  for (const c of chunks) {
    if (buf.length + c.length + 1 <= target) {
      buf = buf ? buf + "\n" + c : c;
    } else {
      if (buf) merged.push(buf);
      buf = c;
    }
  }
  if (buf) merged.push(buf);
  return merged;
}

// Faccette del front-matter da iniettare nel testo indicizzato (entity-aware indexing).
// Rende cercabili doc-type/verticale/tag e — per le note-entità SKOS — prefLabel/altLabel:
// così una query "OP" trova la nota "Ordine di Pagamento" via i suoi sinonimi.
const META_CTX_KEYS = ["doc-type", "tipo", "verticale", "layer", "tags"];

function cleanScalar(v: string): string {
  return v.replace(/^\s*\[/, "").replace(/\]\s*$/, "").replace(/"/g, "").trim();
}

function facetLine(meta: Record<string, string>): string {
  const parts: string[] = [];
  for (const k of META_CTX_KEYS) {
    const v = meta[k];
    if (v && cleanScalar(v)) parts.push(`[${k}: ${cleanScalar(v)}]`);
  }
  return parts.join(" ");
}

/** Chunk-identità: titolo + faccette + alias SKOS + scopeNote. null se non c'è metadata utile. */
function identityCard(meta: Record<string, string>, fileName: string): string | null {
  const bits: string[] = [];
  const title = meta["prefLabel"] ? cleanScalar(meta["prefLabel"]) : fileName.replace(/\.md$/, "");
  bits.push(title);
  const fl = facetLine(meta);
  if (fl) bits.push(fl);
  const aliases = [meta["prefLabel"], meta["altLabel"], meta["hiddenLabel"]]
    .filter(Boolean)
    .map(cleanScalar)
    .filter((s) => s.length > 0);
  if (aliases.length) bits.push("Termini: " + aliases.join(", "));
  if (meta["scopeNote"] && cleanScalar(meta["scopeNote"])) bits.push(cleanScalar(meta["scopeNote"]));
  return bits.length > 1 ? bits.join("\n") : null; // solo se c'è più del titolo
}

export function chunkMarkdown(text: string, fileName: string): Chunk[] {
  const { meta, body } = parseFrontmatter(text);
  const chunks: Chunk[] = [];
  let idx = 0;
  const facets = facetLine(meta);

  // 1) chunk-identità (entity-aware): rende l'header semantico + gli alias cercabili
  const card = identityCard(meta, fileName);
  if (card) {
    chunks.push({ content: `[File: ${fileName}]\n${card}`, heading: "", headerPath: "", chunkIndex: idx });
    idx++;
  }

  // 2) chunk di contenuto, con faccette nel prefisso
  for (const { headerPath, heading, text: secText } of splitSections(body)) {
    let prefix = `[File: ${fileName}]`;
    if (headerPath) prefix += ` [Sezione: ${headerPath}]`;
    if (facets) prefix += ` ${facets}`;
    prefix += "\n";
    const avail = Math.max(200, MAX_CHARS - prefix.length);
    const pieces = mergeSmall(recursiveSplit(secText, avail), Math.max(200, TARGET_CHARS - prefix.length));
    for (const piece of pieces) {
      chunks.push({ content: prefix + piece, heading, headerPath, chunkIndex: idx });
      idx++;
    }
  }
  return chunks;
}
