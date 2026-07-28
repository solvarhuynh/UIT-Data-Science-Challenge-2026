// Domain types for the HCMUTE-SHIPCODE Legal Intelligence API.

export interface LegalDocumentMetadata {
  doc_id: string;
  law_name: string;
  doc_number?: string;
  issued_date?: string;
  status?: "Còn hiệu lực" | "Hết hiệu lực" | "Sửa đổi bổ sung";
  doc_type?: "Bộ luật" | "Luật" | "Nghị định" | "Thông tư" | "Quyết định";
  chapter?: string;
  article?: string;
  clause?: string;
  point?: string;
  source?: string;
}

export interface RetrievalHit {
  chunk_id: string;
  doc_id: string;
  text: string;
  score: number;
  source: string;
  law_name: string;
  chapter?: string;
  article?: string;
  clause?: string;
  point?: string;
  parent_id?: string;
  dense_score: number;
  sparse_score: number;
  hybrid_score: number;
  rerank_score: number;
  final_score: number;
  rank: number;
  metadata: LegalDocumentMetadata;
}

export interface Citation {
  id: string;
  label: string;
  title?: string;
  doc_id?: string;
  law_name: string;
  doc_number?: string;
  article?: string;
  clause?: string;
  point?: string;
  excerpt: string;
  chunk_id: string;
  score: number;
  source: string;
  rank: number;
}

export interface PipelineLatency {
  embed_ms: number;
  retrieve_ms: number;
  rerank_ms: number;
  generate_ms: number;
  total_ms: number;
}

export type QueryMode = "qa" | "lookup" | "compare";

export interface QueryFilters {
  law_name?: string | null;
  doc_id?: string | null;
  article?: string | null;
}

export interface QueryRequest {
  question: string;
  top_k: number;
  top_n: number;
  filters: QueryFilters | null;
  prompt_version: string;
  debug: boolean;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  retrieval_hits: RetrievalHit[];
  latency_ms: PipelineLatency;
  cache_hit: boolean;
  prompt_version: string;
  warnings: string[];
  trace_id: string;
  confidence?: "high" | "medium" | "low";
  mock?: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  mode?: QueryMode;
  response?: QueryResponse;
  followUps?: string[];
  error?: "empty" | "insufficient" | "timeout" | "offline" | "backend";
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  pinned: boolean;
  messages: Message[];
}

export interface RetrievalSettings {
  topK: 5 | 10 | 20;
  lawName: string;
  docId: string;
  article: string;
  advanced: boolean;
}

export interface SavedLegalSource {
  id: string;
  title: string;
  doc_id?: string;
  law_name: string;
  doc_number?: string;
  article?: string;
  clause?: string;
  point?: string;
  excerpt: string;
  source?: string;
  savedAt: number;
}
