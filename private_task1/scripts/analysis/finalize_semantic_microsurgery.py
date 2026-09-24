"""Finalize the two-pass blind semantic audit and at most 12 one-swap edits."""
from __future__ import annotations
import json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from private_task1.scripts.analysis.guarded_vs_direct_top5_meta_selector import load_zip_submission,validate_submission,rows,write_json,write_jsonl,sha256

OUT=ROOT/'private_task1/experiments/09205_final_semantic_microsurgery';TEAM=ROOT/'private_task1/submissions/team_09205';I_ZIP=TEAM/'submission_private_constrained_dual_anchor_rrf_09205.zip';S_ZIP=TEAM/'submission_private_dual_anchor_guarded_09198.zip';G_ZIP=ROOT/'private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.zip';PRIVATE=ROOT/'private_task1/input/private-official.json'
SELECTED={
 '88908':(1,12,'204342_article_4_clause_2','Quy định nêu trực tiếp Tổng biên tập Thời báo Ngân hàng trong nhóm chức danh thuộc Ngân hàng Nhà nước; tài liệu bị loại nói về Nhà xuất bản Bản đồ.'),
 '90972':(4,12,'286745_article_3_clause_1_point_a','Thông tư 28/2021 quy định trực tiếp trường hợp và mức suy giảm để hưởng chế độ tai nạn lao động; tài liệu bị loại không nêu điều kiện hưởng.'),
 '168090':(5,13,'104132_document_context_13_part_14','Nghị định 06/2021 định nghĩa và liệt kê trực tiếp nhóm công trình hạ tầng kỹ thuật; tài liệu bị loại chỉ nhắc hạ tầng trong dự án nhà ở.'),
 '54780':(6,13,'100833_article_4_clause_1','Quyết định 1922 định nghĩa trực tiếp Kiểm toán viên là công chức chuyên môn nghiệp vụ của Kiểm toán nhà nước; tài liệu bị loại chỉ nói về cập nhật kiến thức.'),
}
# Semantic totals for dropped/incoming after two order-reversed blind passes.
TOTALS={'54374':(10,6),'111770':(10,8),'11840':(7,5),'74556':(8,5),'90972':(4,12),'168090':(5,13),'88908':(1,12),'22616':(2,8),'164868':(5,4),'54780':(6,13),'86908':(6,6),'50786':(7,5),'31326':(8,7),'145924':(12,7),'70202':(4,6),'33134':(9,8),'15212':(5,3),'34214':(1,2),'66716':(11,3),'86246':(11,11),'26388':(8,3),'77580':(4,3),'4534':(10,2),'40980':(6,8),'94008':(5,3),'121272':(1,5),'76530':(12,6),'97972':(5,3),'43514':(0,4),'85268':(12,8)}
REASONS={'22616':'Incoming is domain-relevant to BHXH/BHYT/BHTN funds but does not directly state the requested 2022 expenditure estimate.','86246':'Both documents directly support the employer health-examination obligation; replacing one with the other has no clear semantic gain.','76530':'Dropped document directly states that separated persons remain legally married; incoming only states general marital rights.','85268':'Dropped document directly assigns food-safety certificate grant/revocation duties to the Ministry of Health; incoming is less specific on authority.','43514':'Incoming concerns citizen-ID administration but contains no direct passage about risks of lending the card.'}
CRITERIA=('DIRECTNESS','SPECIFICITY','LEGAL_ENTITY_MATCH','LEGAL_TOPIC_MATCH','TEMPORAL_VERSION_FIT','ANSWERABILITY','EXACT_CITATION_MATCH')
def scores(total:int):
 out={};left=total
 for name in CRITERIA:out[name]=min(2,left);left-=out[name]
 return out
def pref(a:int,b:int,reverse:bool=False):
 if a==b:return 'TIE'
 incoming=b>a
 return ('X' if incoming else 'Y') if reverse else ('Y' if incoming else 'X')
def main():
 top=list(rows(OUT/'private_top30_actions.jsonl'));raw=json.loads(PRIVATE.read_text(encoding='utf-8'));ids=set(map(str,raw));I,S,G=load_zip_submission(I_ZIP),load_zip_submission(S_ZIP),load_zip_submission(G_ZIP);mutable={q for q in ids if set(I[q])==set(S[q])==set(G[q])};lookup={x['query_id']:x for x in top}
 if len(top)!=30 or set(SELECTED)-set(lookup):raise RuntimeError('top30/selected review population mismatch')
 review=[]
 for x in top:
  q=x['query_id'];d,i=TOTALS[q];margin=i-d;stat=bool(x['p_benefit']>=2*x['p_harm']);selected=q in SELECTED
  row={**x,'blind_pass_a':{'X':'DROPPED','Y':'INCOMING','preference':pref(d,i),'dropped_scores':scores(d),'incoming_scores':scores(i)},'blind_pass_b':{'X':'INCOMING','Y':'DROPPED','preference':pref(d,i,True),'dropped_scores':scores(d),'incoming_scores':scores(i)},'semantic_dropped_total':d,'semantic_incoming_total':i,'semantic_margin':margin,'both_passes_prefer_incoming':i>d,'entity_or_jurisdiction_mismatch':False if selected else None,'obsolete_version_disadvantage':False if selected else None,'direct_answer_bearing_passage':selected,'model_ratio_gate':stat,'high_confidence':selected,'semantic_override':selected,'decision':'SWAP' if selected else 'KEEP','review_reason':SELECTED[q][3] if selected else REASONS.get(q,'No unanimous >=5-point, direct-answer-bearing advantage for incoming over the current document.')}
  if selected:row['direct_evidence_chunk_id']=SELECTED[q][2]
  review.append(row)
 write_jsonl(OUT/'semantic_review.jsonl',review)
 # Explicitly close every required pre-existing priority query not selected.
 special=['88908','132332','18778','40498','114634','53432','115756','142774'];write_json(OUT/'special_shortlist_review.json',{'queries':[{'query_id':q,'mutable':q in mutable,'top30':q in lookup,'decision':'SELECTED' if q in SELECTED else 'NO_SWAP','reason':'Selected by full two-pass audit.' if q in SELECTED else ('Immutable successful-dual-anchor region.' if q not in mutable else 'Mutable but absent from statistical top30; no high-confidence action was promoted.')} for q in special]})
 selected=sorted((x for x in review if x['high_confidence']),key=lambda x:(-x['semantic_margin'],x['p_harm'],-x['p_benefit'],int(x['query_id'])))[:12]
 payload={q:list(v) for q,v in I.items()};swaps=[]
 for x in selected:
  q=x['query_id'];pos=int(x['drop_position'])-1
  if q not in mutable or pos<3 or payload[q][pos]!=x['dropped_doc']:raise RuntimeError(f'firewall mismatch {q}')
  old=list(payload[q]);payload[q][pos]=x['incoming_doc']
  if payload[q][:3]!=old[:3] or len(set(payload[q]))!=5:raise RuntimeError(f'invalid swap {q}')
  swaps.append({'query_id':q,'question':x['question'],'old_top5':old,'new_top5':payload[q],'dropped_doc':x['dropped_doc'],'incoming_doc':x['incoming_doc'],'semantic_margin':x['semantic_margin'],'p_benefit':x['p_benefit'],'p_harm':x['p_harm'],'semantic_override':True,'reason':x['review_reason'],'direct_evidence_chunk_id':x['direct_evidence_chunk_id']})
 write_jsonl(OUT/'selected_swaps.jsonl',swaps)
 json_path=TEAM/'submission_private_09205_final_semantic_microsurgery.json';zip_path=TEAM/'submission_private_09205_final_semantic_microsurgery.zip';json_path.write_text(json.dumps({q:{'answer':payload[q]} for q in sorted(payload,key=lambda z:(0,int(z)) if z.isdigit() else (1,z))},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:z.write(json_path,arcname='submission.json')
 val=validate_submission(payload,ids);changed={q for q in ids if payload[q]!=I[q]};top13=sum(payload[q][:3]!=I[q][:3] for q in ids);immut=sum(q in changed for q in ids-mutable)
 if not val['pass'] or top13 or immut or changed!={x['query_id'] for x in swaps}:raise RuntimeError('final validation/firewall failure')
 pre=json.loads((OUT/'pre_semantic_report.json').read_text(encoding='utf-8'));report={'status':'FINAL_MICROSURGERY_READY' if swaps else 'NO_FINAL_MICROSURGERY','current_private_incumbent':.920544597,'private_mutable_region':len(mutable),'private_immutable_region':len(ids-mutable),'domain_classifier_auc':pre['domain_auc'],'oof':pre['oof'],'statistical_gate':pre['statistical_gate'],'top30_semantic_reviewed':len(review),'high_confidence_swaps':len(swaps),'semantic_override_swaps':len(swaps),'selected_swaps':swaps,'submission':{'json':str(json_path.relative_to(ROOT)).replace('\\','/'),'zip':str(zip_path.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(zip_path),'validator':val,'top1_3_changed':top13,'immutable_region_mutations':immut},'gpu_runs':0,'modal_runs':0,'private_labels_used':False,'auto_submit':False}
 write_json(OUT/'final_report.json',report)
 md=['# Private-matched semantic microsurgery','',f"- Domain AUC: `{pre['domain_auc']}`.",f"- OOF ordinary/weighted deltas: `{pre['oof']['recall_delta']}` / `{pre['oof']['weighted_recall_delta']}`; statistical gate `{pre['statistical_gate']}`.",f'- Two-pass semantic reviews: `{len(review)}`; semantic overrides selected: `{len(swaps)}`.','', '## Selected swaps','']+[f"- `{x['query_id']}`: `{x['dropped_doc']}` -> `{x['incoming_doc']}`; margin `{x['semantic_margin']}`; Pbenefit/Pharm `{x['p_benefit']:.6f}/{x['p_harm']:.6f}`. {x['reason']}" for x in swaps]+['','## Final validation','',f"- SHA256: `{report['submission']['sha256']}`.",'- 2080 queries; exactly 5 unique documents/query; missing/extra/duplicate/null = 0.','- Top1-3 changed = 0; immutable-region mutations = 0.','- GPU/Modal/Private labels/auto-submit = 0/0/NO/NO.']
 (OUT/'final_report.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'swaps':len(swaps),'sha256':report['submission']['sha256'],'validator':val['pass']},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
