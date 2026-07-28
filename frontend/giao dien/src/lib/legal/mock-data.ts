import type {
  Citation,
  Conversation,
  LegalDocumentMetadata,
  Message,
  QueryResponse,
  RetrievalHit,
} from "./types";

const now = Date.now();

export const MOCK_HITS: RetrievalHit[] = [
  {
    chunk_id: "bllđ2019_d35_k1_c0",
    doc_id: "45/2019/QH14",
    text: "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động ít nhất 45 ngày nếu làm việc theo hợp đồng lao động không xác định thời hạn; ít nhất 30 ngày nếu làm việc theo hợp đồng lao động xác định thời hạn có thời hạn từ 12 tháng đến 36 tháng; ít nhất 03 ngày làm việc nếu làm việc theo hợp đồng lao động xác định thời hạn có thời hạn dưới 12 tháng.",
    score: 0.94,
    source: "data/corpus/bo-luat-lao-dong-2019/chuong-III/dieu-35.md",
    law_name: "Bộ luật Lao động 2019",
    chapter: "Chương III — Hợp đồng lao động",
    article: "Điều 35",
    clause: "Khoản 1",
    parent_id: "bllđ2019_d35",
    dense_score: 0.912,
    sparse_score: 0.771,
    hybrid_score: 0.883,
    rerank_score: 0.958,
    final_score: 0.941,
    rank: 1,
    metadata: {
      doc_id: "45/2019/QH14",
      law_name: "Bộ luật Lao động 2019",
      doc_number: "45/2019/QH14",
      issued_date: "20/11/2019",
      status: "Còn hiệu lực",
      doc_type: "Bộ luật",
      chapter: "Chương III",
      article: "Điều 35",
      clause: "Khoản 1",
      source: "Quốc hội khóa XIV",
    },
  },
  {
    chunk_id: "bllđ2019_d35_k2_c0",
    doc_id: "45/2019/QH14",
    text: "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động không cần báo trước trong trường hợp: không được bố trí theo đúng công việc, địa điểm làm việc hoặc không được bảo đảm điều kiện làm việc theo thỏa thuận; không được trả đủ lương hoặc trả lương không đúng thời hạn; bị người sử dụng lao động ngược đãi, đánh đập hoặc có lời nói, hành vi nhục mạ, hành vi làm ảnh hưởng đến sức khỏe, nhân phẩm, danh dự; bị quấy rối tình dục tại nơi làm việc; lao động nữ mang thai phải nghỉ việc theo quy định; đủ tuổi nghỉ hưu, trừ trường hợp các bên có thỏa thuận khác; người sử dụng lao động cung cấp thông tin không trung thực làm ảnh hưởng đến việc thực hiện hợp đồng lao động.",
    score: 0.91,
    source: "data/corpus/bo-luat-lao-dong-2019/chuong-III/dieu-35.md",
    law_name: "Bộ luật Lao động 2019",
    chapter: "Chương III — Hợp đồng lao động",
    article: "Điều 35",
    clause: "Khoản 2",
    point: "Điểm a–g",
    parent_id: "bllđ2019_d35",
    dense_score: 0.889,
    sparse_score: 0.802,
    hybrid_score: 0.871,
    rerank_score: 0.933,
    final_score: 0.912,
    rank: 2,
    metadata: {
      doc_id: "45/2019/QH14",
      law_name: "Bộ luật Lao động 2019",
      doc_number: "45/2019/QH14",
      issued_date: "20/11/2019",
      status: "Còn hiệu lực",
      doc_type: "Bộ luật",
      chapter: "Chương III",
      article: "Điều 35",
      clause: "Khoản 2",
      source: "Quốc hội khóa XIV",
    },
  },
  {
    chunk_id: "bllđ2019_d40_c0",
    doc_id: "45/2019/QH14",
    text: "Người lao động đơn phương chấm dứt hợp đồng lao động trái pháp luật thì không được trợ cấp thôi việc; phải bồi thường cho người sử dụng lao động nửa tháng tiền lương theo hợp đồng lao động và một khoản tiền tương ứng với tiền lương theo hợp đồng lao động trong những ngày không báo trước; phải hoàn trả cho người sử dụng lao động chi phí đào tạo.",
    score: 0.83,
    source: "data/corpus/bo-luat-lao-dong-2019/chuong-III/dieu-40.md",
    law_name: "Bộ luật Lao động 2019",
    chapter: "Chương III — Hợp đồng lao động",
    article: "Điều 40",
    parent_id: "bllđ2019_d40",
    dense_score: 0.815,
    sparse_score: 0.744,
    hybrid_score: 0.798,
    rerank_score: 0.851,
    final_score: 0.834,
    rank: 3,
    metadata: {
      doc_id: "45/2019/QH14",
      law_name: "Bộ luật Lao động 2019",
      doc_number: "45/2019/QH14",
      issued_date: "20/11/2019",
      status: "Còn hiệu lực",
      doc_type: "Bộ luật",
      chapter: "Chương III",
      article: "Điều 40",
      source: "Quốc hội khóa XIV",
    },
  },
  {
    chunk_id: "nd145_2020_d7_c0",
    doc_id: "145/2020/NĐ-CP",
    text: "Thời hạn báo trước khi đơn phương chấm dứt hợp đồng lao động đối với một số ngành, nghề, công việc đặc thù là ít nhất 120 ngày đối với hợp đồng lao động không xác định thời hạn hoặc hợp đồng lao động xác định thời hạn từ 12 tháng trở lên.",
    score: 0.72,
    source: "data/corpus/nghi-dinh-145-2020/dieu-7.md",
    law_name: "Nghị định 145/2020/NĐ-CP",
    chapter: "Chương II",
    article: "Điều 7",
    parent_id: "nd145_d7",
    dense_score: 0.703,
    sparse_score: 0.681,
    hybrid_score: 0.697,
    rerank_score: 0.741,
    final_score: 0.722,
    rank: 4,
    metadata: {
      doc_id: "145/2020/NĐ-CP",
      law_name: "Nghị định 145/2020/NĐ-CP",
      doc_number: "145/2020/NĐ-CP",
      issued_date: "14/12/2020",
      status: "Còn hiệu lực",
      doc_type: "Nghị định",
      chapter: "Chương II",
      article: "Điều 7",
      source: "Chính phủ",
    },
  },
  {
    chunk_id: "bllđ2019_d113_k1_c0",
    doc_id: "45/2019/QH14",
    text: "Người lao động làm việc đủ 12 tháng cho một người sử dụng lao động thì được nghỉ hằng năm, hưởng nguyên lương theo hợp đồng lao động 12 ngày làm việc đối với người làm công việc trong điều kiện bình thường.",
    score: 0.61,
    source: "data/corpus/bo-luat-lao-dong-2019/chuong-VII/dieu-113.md",
    law_name: "Bộ luật Lao động 2019",
    chapter: "Chương VII — Thời giờ làm việc, thời giờ nghỉ ngơi",
    article: "Điều 113",
    clause: "Khoản 1",
    parent_id: "bllđ2019_d113",
    dense_score: 0.598,
    sparse_score: 0.532,
    hybrid_score: 0.581,
    rerank_score: 0.632,
    final_score: 0.611,
    rank: 5,
    metadata: {
      doc_id: "45/2019/QH14",
      law_name: "Bộ luật Lao động 2019",
      doc_number: "45/2019/QH14",
      issued_date: "20/11/2019",
      status: "Còn hiệu lực",
      doc_type: "Bộ luật",
      chapter: "Chương VII",
      article: "Điều 113",
      clause: "Khoản 1",
      source: "Quốc hội khóa XIV",
    },
  },
];

export const MOCK_CITATIONS: Citation[] = MOCK_HITS.slice(0, 3).map((h) => ({
  id: `c${h.rank}`,
  label: [h.law_name, h.article, h.clause].filter(Boolean).join(", "),
  doc_id: h.doc_id,
  law_name: h.law_name,
  doc_number: h.metadata.doc_number,
  article: h.article,
  clause: h.clause,
  point: h.point,
  excerpt: h.text,
  chunk_id: h.chunk_id,
  score: h.final_score,
  source: h.source,
  rank: h.rank,
}));

export const MOCK_ANSWER = `Người lao động **có quyền đơn phương chấm dứt hợp đồng lao động** nhưng phải tuân thủ thời hạn báo trước, trừ một số trường hợp pháp luật cho phép không cần báo trước.

### Thời hạn báo trước
- Ít nhất **45 ngày** đối với hợp đồng lao động không xác định thời hạn [[c1]]
- Ít nhất **30 ngày** đối với hợp đồng xác định thời hạn từ 12 tháng đến 36 tháng [[c1]]
- Ít nhất **03 ngày làm việc** đối với hợp đồng có thời hạn dưới 12 tháng [[c1]]

Đối với một số ngành, nghề, công việc đặc thù, thời hạn báo trước có thể dài hơn theo quy định riêng của Chính phủ.

### Trường hợp không cần báo trước
Người lao động được chấm dứt hợp đồng ngay, không phải báo trước, khi thuộc một trong các trường hợp sau [[c2]]:
- Không được bố trí đúng công việc, địa điểm hoặc điều kiện làm việc như đã thỏa thuận
- Không được trả đủ lương hoặc trả lương không đúng thời hạn
- Bị ngược đãi, đánh đập, nhục mạ hoặc bị quấy rối tình dục tại nơi làm việc
- Lao động nữ mang thai phải nghỉ việc theo chỉ định của cơ sở khám bệnh, chữa bệnh
- Đủ tuổi nghỉ hưu, trừ trường hợp các bên có thỏa thuận khác

### Hệ quả khi vi phạm thời hạn báo trước
Việc chấm dứt hợp đồng trái quy định khiến người lao động **không được trợ cấp thôi việc**, phải bồi thường nửa tháng tiền lương cùng khoản tiền tương ứng tiền lương những ngày không báo trước, và hoàn trả chi phí đào tạo (nếu có) [[c3]].`;

export const MOCK_RESPONSE: QueryResponse = {
  answer: MOCK_ANSWER,
  citations: MOCK_CITATIONS,
  retrieval_hits: MOCK_HITS,
  latency_ms: {
    embed_ms: 42,
    retrieve_ms: 318,
    rerank_ms: 264,
    generate_ms: 1712,
    total_ms: 2336,
  },
  cache_hit: false,
  prompt_version: "legal_qa_v1",
  warnings: [],
  trace_id: "trc_8f31c2a9d0e4",
  confidence: "high",
  mock: true,
};

export const FOLLOW_UPS = [
  "Trường hợp nào không cần báo trước?",
  "Nếu vi phạm thời hạn báo trước thì sao?",
  "Quy định này áp dụng cho hợp đồng thử việc không?",
];

export const SUGGESTED_PROMPTS = [
  "Điều kiện đơn phương chấm dứt hợp đồng lao động?",
  "Người lao động được nghỉ phép bao nhiêu ngày?",
  "So sánh hợp đồng xác định và không xác định thời hạn",
  "Thủ tục đăng ký thành lập doanh nghiệp gồm những gì?",
];

export const SAMPLE_QUESTION =
  "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động trong trường hợp nào?";

export function buildSampleConversation(): Conversation {
  const messages: Message[] = [
    {
      id: "m1",
      role: "user",
      content: SAMPLE_QUESTION,
      createdAt: now - 1000 * 60 * 12,
      mode: "qa",
    },
    {
      id: "m2",
      role: "assistant",
      content: MOCK_ANSWER,
      createdAt: now - 1000 * 60 * 11,
      response: MOCK_RESPONSE,
      followUps: FOLLOW_UPS,
    },
  ];
  return {
    id: "conv-sample",
    title: "Điều kiện chấm dứt hợp đồng",
    createdAt: now - 1000 * 60 * 12,
    updatedAt: now - 1000 * 60 * 11,
    pinned: false,
    messages,
  };
}

export function buildSeedConversations(): Conversation[] {
  return [buildSampleConversation()];
}

export interface LegalDocEntry extends LegalDocumentMetadata {
  title: string;
  hierarchy: string[];
  excerpt: string;
  year: number;
}

export const LEGAL_DOCS: LegalDocEntry[] = [
  {
    doc_id: "45/2019/QH14",
    law_name: "Bộ luật Lao động 2019",
    doc_number: "45/2019/QH14",
    doc_type: "Bộ luật",
    status: "Còn hiệu lực",
    issued_date: "20/11/2019",
    year: 2019,
    article: "Điều 35",
    clause: "Khoản 1",
    title: "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
    hierarchy: ["Bộ luật Lao động 2019", "Chương III", "Điều 35", "Khoản 1"],
    excerpt:
      "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động ít nhất 45 ngày…",
    source: "data/corpus/bo-luat-lao-dong-2019/chuong-III/dieu-35.md",
  },
  {
    doc_id: "45/2019/QH14",
    law_name: "Bộ luật Lao động 2019",
    doc_number: "45/2019/QH14",
    doc_type: "Bộ luật",
    status: "Còn hiệu lực",
    issued_date: "20/11/2019",
    year: 2019,
    article: "Điều 113",
    clause: "Khoản 1",
    title: "Nghỉ hằng năm",
    hierarchy: ["Bộ luật Lao động 2019", "Chương VII", "Điều 113", "Khoản 1"],
    excerpt:
      "Người lao động làm việc đủ 12 tháng cho một người sử dụng lao động thì được nghỉ hằng năm, hưởng nguyên lương 12 ngày làm việc…",
    source: "data/corpus/bo-luat-lao-dong-2019/chuong-VII/dieu-113.md",
  },
  {
    doc_id: "59/2020/QH14",
    law_name: "Luật Doanh nghiệp 2020",
    doc_number: "59/2020/QH14",
    doc_type: "Luật",
    status: "Còn hiệu lực",
    issued_date: "17/06/2020",
    year: 2020,
    article: "Điều 26",
    title: "Trình tự, thủ tục đăng ký doanh nghiệp",
    hierarchy: ["Luật Doanh nghiệp 2020", "Chương II", "Điều 26"],
    excerpt:
      "Người thành lập doanh nghiệp nộp hồ sơ đăng ký doanh nghiệp tại Cơ quan đăng ký kinh doanh; Cơ quan đăng ký kinh doanh cấp Giấy chứng nhận đăng ký doanh nghiệp trong thời hạn 03 ngày làm việc…",
    source: "data/corpus/luat-doanh-nghiep-2020/dieu-26.md",
  },
  {
    doc_id: "145/2020/NĐ-CP",
    law_name: "Nghị định 145/2020/NĐ-CP",
    doc_number: "145/2020/NĐ-CP",
    doc_type: "Nghị định",
    status: "Còn hiệu lực",
    issued_date: "14/12/2020",
    year: 2020,
    article: "Điều 7",
    title: "Thời hạn báo trước đối với ngành, nghề đặc thù",
    hierarchy: ["Nghị định 145/2020/NĐ-CP", "Chương II", "Điều 7"],
    excerpt:
      "Thời hạn báo trước khi đơn phương chấm dứt hợp đồng lao động đối với một số ngành, nghề, công việc đặc thù là ít nhất 120 ngày…",
    source: "data/corpus/nghi-dinh-145-2020/dieu-7.md",
  },
  {
    doc_id: "91/2015/QH13",
    law_name: "Bộ luật Dân sự 2015",
    doc_number: "91/2015/QH13",
    doc_type: "Bộ luật",
    status: "Còn hiệu lực",
    issued_date: "24/11/2015",
    year: 2015,
    article: "Điều 385",
    title: "Khái niệm hợp đồng",
    hierarchy: ["Bộ luật Dân sự 2015", "Phần thứ ba", "Điều 385"],
    excerpt:
      "Hợp đồng là sự thỏa thuận giữa các bên về việc xác lập, thay đổi hoặc chấm dứt quyền, nghĩa vụ dân sự.",
    source: "data/corpus/bo-luat-dan-su-2015/dieu-385.md",
  },
  {
    doc_id: "58/2020/QH14",
    law_name: "Luật Bảo hiểm xã hội 2014",
    doc_number: "58/2014/QH13",
    doc_type: "Luật",
    status: "Sửa đổi bổ sung",
    issued_date: "20/11/2014",
    year: 2014,
    article: "Điều 60",
    title: "Bảo hiểm xã hội một lần",
    hierarchy: ["Luật Bảo hiểm xã hội 2014", "Chương III", "Điều 60"],
    excerpt:
      "Người lao động có yêu cầu thì được hưởng bảo hiểm xã hội một lần nếu thuộc một trong các trường hợp quy định tại khoản 1 Điều này.",
    source: "data/corpus/luat-bhxh-2014/dieu-60.md",
  },
];
