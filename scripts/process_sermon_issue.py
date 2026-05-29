import datetime as dt
import json, os, random, re, sys, time
from pathlib import Path
from github import Github, GithubException
from google import genai
from google.genai import types
from slugify import slugify

GEMINI_API_KEY=os.environ["GEMINI_API_KEY"]
GITHUB_TOKEN=os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY=os.environ["GITHUB_REPOSITORY"]
RUN_MODE=os.environ.get("RUN_MODE","single")
ISSUE_NUMBER_ENV=os.environ.get("ISSUE_NUMBER","").strip()
RETRY_LIMIT_PER_RUN=int(os.environ.get("RETRY_LIMIT_PER_RUN","2"))
GEMINI_MODELS=[m.strip() for m in os.environ.get("GEMINI_MODELS","gemini-2.5-flash-lite,gemini-2.5-flash").split(",") if m.strip()]
MAX_RETRY_ATTEMPTS=12
LABELS={
 "request":("sermon-request","ededed"),"processing":("processing","fbca04"),
 "retry":("retry-needed","d93f0b"),"done":("done","0e8a16"),"failed":("failed","b60205")}

def gh_repo(): return Github(GITHUB_TOKEN).get_repo(GITHUB_REPOSITORY)
def ensure_labels(repo):
    existing={l.name for l in repo.get_labels()}
    for name,color in LABELS.values():
        if name not in existing:
            try: repo.create_label(name=name,color=color)
            except GithubException: pass

def label_names(issue): return {l.name for l in issue.get_labels()}
def add_labels(issue,*names):
    cur=label_names(issue)
    for n in names:
        if n and n not in cur: issue.add_to_labels(n)
def remove_labels(issue,*names):
    cur=label_names(issue)
    for n in names:
        if n and n in cur: issue.remove_from_labels(n)

def korea_today(): return (dt.datetime.utcnow()+dt.timedelta(hours=9)).date().isoformat()
def issue_value(body,label):
    m=re.search(rf"### {re.escape(label)}\s*\n\s*(.*?)(?=\n### |\Z)", body or "", re.DOTALL)
    if not m: return ""
    v=m.group(1).strip()
    return "" if v in {"_No response_","No response"} else v

def to_int(v,default,lo,hi):
    try: n=int(str(v).strip())
    except Exception: n=default
    return max(lo,min(n,hi))
def mmss(sec): return f"{sec//60:02d}:{sec%60:02d}"
def split_text(text,max_chars=9000):
    text=(text or "").strip()
    if len(text)<=max_chars: return [text] if text else []
    chunks=[]; start=0
    while start<len(text):
        end=min(start+max_chars,len(text)); b=text.rfind("\n\n",start,end)
        if b==-1 or b<=start+1000: b=text.rfind(". ",start,end)
        if b==-1 or b<=start+1000: b=end
        chunks.append(text[start:b].strip()); start=b
    return [c for c in chunks if c]

def safe_json(text):
    text=(text or "").strip(); text=re.sub(r"^```json\s*","",text); text=re.sub(r"^```\s*","",text); text=re.sub(r"\s*```$","",text).strip()
    try: return json.loads(text)
    except Exception:
        m=re.search(r"\{.*\}",text,re.DOTALL)
        if m: return json.loads(m.group(0))
    raise RuntimeError("Gemini 분석 결과를 JSON으로 해석하지 못했습니다.")

def retryable_error(e):
    s=str(e).lower()
    return any(x in s for x in ["503","unavailable","high demand","429","resource_exhausted","try again later","timeout","deadline"])

def client(): return genai.Client(api_key=GEMINI_API_KEY)
def call_gemini(fn,purpose):
    last=None
    for model in GEMINI_MODELS:
        for attempt in range(1,5):
            try:
                print(f"{purpose}: {model}, attempt {attempt}", flush=True)
                return fn(model)
            except Exception as e:
                last=e
                if not retryable_error(e): raise
                sleep=15*(2**(attempt-1))+random.randint(0,6)
                print(f"retry after {sleep}s: {e}", flush=True); time.sleep(sleep)
    raise RuntimeError(f"Gemini 요청 실패: {last}")

def youtube_request(url,prompt,start,end):
    c=client(); part=types.Part(file_data=types.FileData(file_uri=url), video_metadata=types.VideoMetadata(start_offset=f"{int(start)}s", end_offset=f"{int(end)}s"))
    def run(model):
        r=c.models.generate_content(model=model, contents=types.Content(parts=[part, types.Part(text=prompt)]))
        return (r.text or "").strip()
    return call_gemini(run,"YouTube 구간 전사")

def text_request(prompt,purpose):
    c=client()
    def run(model):
        r=c.models.generate_content(model=model, contents=prompt)
        return (r.text or "").strip()
    return call_gemini(run,purpose)

def transcribe_from_youtube(url,max_minutes,segment_minutes):
    chunks=[]; total=(max_minutes+segment_minutes-1)//segment_minutes
    for i in range(total):
        start=i*segment_minutes*60; end=min((i+1)*segment_minutes*60,max_minutes*60); s,e=mmss(start),mmss(end)
        prompt=f"""
당신은 한국어 기독교 설교 전문 전사자입니다.
이 YouTube 영상의 {s}~{e} 구간에서 들리는 설교 음성을 가능한 한 충실하게 한국어 문장으로 전사하십시오.
- 요약하지 마십시오.
- 해설을 붙이지 마십시오.
- 영상 분석 설명을 하지 마십시오.
- 이 구간이 영상 길이를 넘어가거나 실제 음성이 없으면 정확히 [END]라고만 출력하십시오.
- 결과는 전사문 본문만 출력하십시오.
"""
        t=youtube_request(url,prompt,start,end)
        if not t or "[END]" in t[:80].upper(): break
        chunks.append(f"[{s}~{e}]\n{t.strip()}")
    raw="\n\n".join(chunks).strip()
    if len(raw)<200: raise RuntimeError("전사 결과가 너무 짧습니다. 영상 접근 제한 또는 Gemini 처리 실패일 수 있습니다.")
    return raw

def clean_manual_transcript(text):
    text=(text or "").strip(); text=re.sub(r"(?m)^\s*\d{1,2}:\d{2}(?::\d{2})?\s*$","",text); text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()

def correct(raw):
    out=[]; chunks=split_text(raw)
    for i,chunk in enumerate(chunks,1):
        prompt=f"""
다음 한국어 기독교 설교 전사문을 오타, 띄어쓰기, 문장부호, 성경 인명/지명/본문 표기만 자연스럽게 바로잡으십시오.
원문의 흐름과 설교자의 어투를 보존하고, 요약하거나 내용을 추가하지 마십시오.
구간 표시는 있으면 유지하십시오. 결과는 수정된 설교문 본문만 출력하십시오.

전사문 조각 {i}/{len(chunks)}:
{chunk}
"""
        out.append(text_request(prompt,f"오타 수정 {i}/{len(chunks)}"))
    return "\n\n".join(out).strip()

def analyze(corrected,meta):
    prompt=f"""
다음 설교문을 설교 아카이브용으로 분석하십시오. 반드시 JSON 객체만 출력하십시오.
JSON 형식:
{{"title":"설교 제목","speaker":"설교자","date":"YYYY-MM-DD 또는 미상","bibleText":"본문","church":"교회 또는 채널","summary":"설교 전체 요약. 800~1200자 정도.","mainMessage":"핵심 메시지 한 문단","outline":[{{"title":"대지 제목","summary":"해당 대지 요약"}}],"topics":["주제색인1"],"applications":["적용점1"],"illustrations":[{{"title":"예화 제목","summary":"예화 요약","topics":["연결 주제"]}}]}}
사용자 입력:
- 설교 제목: {meta.get('title') or '미입력'}
- 설교자: {meta.get('speaker') or '미입력'}
- 본문: {meta.get('bibleText') or '미입력'}
- 교회/채널: {meta.get('church') or '미입력'}
- 설교 날짜: {meta.get('date') or '미입력'}
- 유튜브 주소: {meta.get('youtubeUrl')}
- 원문 수집 방식: {meta.get('sourceType')}
설교문:
{corrected[:70000]}
"""
    data=safe_json(text_request(prompt,"설교 분석"))
    for k in ["title","speaker","bibleText","church","date"]:
        if meta.get(k): data[k]=meta[k]
    data.setdefault("title","제목 미상"); data.setdefault("speaker","설교자 미상"); data.setdefault("bibleText","본문 미상"); data.setdefault("church","미상")
    if not data.get("date") or data.get("date")=="미상": data["date"]=korea_today()
    for k in ["outline","topics","applications","illustrations"]:
        if not isinstance(data.get(k),list): data[k]=[]
    data["sourceType"]=meta.get("sourceType") or "github_actions"
    return data

def sermon_id(date,title): return f"{date}-{slugify(title or 'sermon', lowercase=True) or 'sermon'}"[:120].strip("-")
def card_md(item):
    outline="\n".join(f"{i+1}. **{o.get('title','대지')}** — {o.get('summary','')}" for i,o in enumerate(item.get("outline",[]))) or "대지 정보가 없습니다."
    apps="\n".join(f"- {a}" for a in item.get("applications",[])) or "적용점 정보가 없습니다."
    ills="\n".join(f"- **{x.get('title','예화')}**: {x.get('summary','')}" for x in item.get("illustrations",[])) or "추출된 예화가 없습니다."
    topics=", ".join(item.get("topics",[])) or "주제 정보가 없습니다."
    return f"""# {item.get('title','제목 미상')}

- 설교자: {item.get('speaker','설교자 미상')}
- 날짜: {item.get('date','날짜 미상')}
- 본문: {item.get('bibleText','본문 미상')}
- 교회/채널: {item.get('church','미상')}
- 영상: {item.get('videoUrl','')}
- 원문 수집 방식: {item.get('sourceType','')}

## 핵심 메시지

{item.get('mainMessage','')}

## 설교 요약

{item.get('summary','')}

## 대지

{outline}

## 적용점

{apps}

## 주제 색인

{topics}

## 예화

{ills}
"""

def save(raw,corrected,analysis,meta,issue_number):
    date,title=analysis.get("date") or korea_today(), analysis.get("title") or "제목 미상"; sid=sermon_id(date,title); base=Path("sermons")/sid; base.mkdir(parents=True, exist_ok=True)
    item={"id":sid,"videoUrl":meta.get("youtubeUrl",""),"title":title,"speaker":analysis.get("speaker","설교자 미상"),"date":date,"bibleText":analysis.get("bibleText","본문 미상"),"church":analysis.get("church","미상"),"summary":analysis.get("summary",""),"mainMessage":analysis.get("mainMessage",""),"outline":analysis.get("outline",[]),"topics":analysis.get("topics",[]),"applications":analysis.get("applications",[]),"illustrations":analysis.get("illustrations",[]),"sourceType":analysis.get("sourceType",""),"createdAt":dt.datetime.now(dt.timezone.utc).isoformat(),"issueNumber":issue_number,"files":{"rawTranscript":f"sermons/{sid}/raw_transcript.md","correctedTranscript":f"sermons/{sid}/corrected_transcript.md","analysis":f"sermons/{sid}/analysis.json","illustrations":f"sermons/{sid}/illustrations.json","sermonCard":f"sermons/{sid}/sermon_card.md"}}
    (base/"raw_transcript.md").write_text(raw,encoding="utf-8"); (base/"corrected_transcript.md").write_text(corrected,encoding="utf-8"); (base/"analysis.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding="utf-8"); (base/"illustrations.json").write_text(json.dumps(item["illustrations"],ensure_ascii=False,indent=2),encoding="utf-8"); (base/"sermon_card.md").write_text(card_md(item),encoding="utf-8")
    Path("data").mkdir(exist_ok=True); idx=Path("data/sermons.json"); sermons=[]
    if idx.exists():
        try:
            old=json.loads(idx.read_text(encoding="utf-8")); sermons=old.get("sermons", old if isinstance(old,list) else [])
        except Exception: sermons=[]
    replaced=False
    for i,old in enumerate(sermons):
        if old.get("id")==item["id"] or old.get("videoUrl")==item["videoUrl"]: sermons[i]=item; replaced=True; break
    if not replaced: sermons.insert(0,item)
    idx.write_text(json.dumps({"sermons":sermons},ensure_ascii=False,indent=2),encoding="utf-8")
    return item

def count_retry_comments(issue):
    return sum(1 for c in issue.get_comments() if "Gemini 서버 혼잡으로 자동 재시도 대기" in (c.body or ""))
def build_meta(issue):
    b=issue.body or ""
    return {"youtubeUrl":issue_value(b,"유튜브 주소"),"title":issue_value(b,"설교 제목"),"speaker":issue_value(b,"설교자"),"bibleText":issue_value(b,"본문"),"church":issue_value(b,"교회/채널"),"date":issue_value(b,"설교 날짜"),"manualTranscript":clean_manual_transcript(issue_value(b,"비상용 설교 스크립트 붙여넣기")),"maxMinutes":to_int(issue_value(b,"최대 처리 시간(분)"),15,5,180),"segmentMinutes":to_int(issue_value(b,"구간 길이(분)"),5,3,20),"memo":issue_value(b,"메모")}

def process_issue(repo,issue):
    ensure_labels(repo); labels=label_names(issue)
    if LABELS["done"][0] in labels: print(f"Issue #{issue.number} already done"); return True
    add_labels(issue,LABELS["request"][0],LABELS["processing"][0]); remove_labels(issue,LABELS["retry"][0],LABELS["failed"][0])
    meta=build_meta(issue); url=meta.get("youtubeUrl","")
    if not url:
        add_labels(issue,LABELS["failed"][0]); remove_labels(issue,LABELS["processing"][0]); issue.create_comment("❌ 유튜브 주소를 찾지 못했습니다. Issue 양식을 확인해 주세요."); return False
    try:
        manual=meta.get("manualTranscript","")
        if manual and len(manual)>=100:
            meta["sourceType"]="manual_transcript_from_issue"; issue.create_comment(f"설교 아카이브 처리를 시작합니다.\n\n- 방식: 비상용 설교 스크립트 붙여넣기\n- 입력 분량: 약 {len(manual):,}자\n\n유튜브 영상 직접 처리는 건너뜁니다."); raw=manual
        else:
            meta["sourceType"]="github_actions_gemini_youtube_url"; issue.create_comment(f"설교 아카이브 처리를 시작합니다.\n\n- 방식: 유튜브 주소 자동 처리\n- 최대 처리 시간: {meta['maxMinutes']}분\n- 구간 길이: {meta['segmentMinutes']}분"); raw=transcribe_from_youtube(url,meta["maxMinutes"],meta["segmentMinutes"])
        corrected=correct(raw); analysis=analyze(corrected,meta); item=save(raw,corrected,analysis,meta,issue.number)
        remove_labels(issue,LABELS["processing"][0],LABELS["retry"][0],LABELS["failed"][0]); add_labels(issue,LABELS["done"][0])
        issue.create_comment(f"✅ 설교 아카이브 생성이 완료되었습니다.\n\n- 제목: {item['title']}\n- 설교자: {item['speaker']}\n- 본문: {item['bibleText']}\n- 저장 ID: `{item['id']}`\n- 처리 방식: {item['sourceType']}\n\n잠시 후 GitHub Pages에서 확인할 수 있습니다.")
        try: issue.edit(state="closed")
        except Exception: pass
        return True
    except Exception as e:
        remove_labels(issue,LABELS["processing"][0])
        if retryable_error(e):
            retry_count=count_retry_comments(issue)+1
            if retry_count>=MAX_RETRY_ATTEMPTS:
                add_labels(issue,LABELS["failed"][0]); remove_labels(issue,LABELS["retry"][0]); issue.create_comment(f"❌ 여러 번 자동 재시도했지만 Gemini 서버 혼잡 오류가 계속되어 중단합니다.\n\n- 재시도 횟수: {retry_count}\n- 나중에 Issue를 다시 편집하거나 Actions에서 수동 재실행할 수 있습니다.\n- 비상용 스크립트 붙여넣기 칸을 사용하면 영상 처리를 건너뛸 수 있습니다."); return False
            add_labels(issue,LABELS["retry"][0]); issue.create_comment(f"⏳ Gemini 서버 혼잡으로 자동 재시도 대기 상태로 전환합니다.\n\n- 자동 재시도 대기 횟수: {retry_count}/{MAX_RETRY_ATTEMPTS}\n- 이 작업은 실패로 종료하지 않고, 약 1시간 뒤 자동 재시도 워크플로에서 다시 처리됩니다.\n- 목사님이 따로 할 일은 없습니다."); print(f"Retryable error on #{issue.number}: {e}"); return True
        add_labels(issue,LABELS["failed"][0]); issue.create_comment(f"❌ 설교 아카이브 생성 중 오류가 발생했습니다.\n\n```text\n{e}\n```"); print(f"Non-retryable error on #{issue.number}: {e}"); return False

def retry_pending(repo):
    ensure_labels(repo); issues=list(repo.get_issues(state="open", labels=[LABELS["retry"][0]])); print(f"Found {len(issues)} retry-needed issues")
    processed=0; all_ok=True
    for issue in issues:
        if processed>=RETRY_LIMIT_PER_RUN: break
        if LABELS["done"][0] in label_names(issue): continue
        print(f"Retrying issue #{issue.number}: {issue.title}"); ok=process_issue(repo,issue); all_ok=all_ok and ok; processed+=1
    return all_ok

def main():
    repo=gh_repo(); ensure_labels(repo)
    if RUN_MODE=="retry_pending": sys.exit(0 if retry_pending(repo) else 1)
    if not ISSUE_NUMBER_ENV: raise RuntimeError("ISSUE_NUMBER가 없습니다.")
    issue=repo.get_issue(number=int(ISSUE_NUMBER_ENV)); sys.exit(0 if process_issue(repo,issue) else 1)
if __name__=="__main__": main()
