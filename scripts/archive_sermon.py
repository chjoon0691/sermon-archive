#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YouTube sermon archive generator for GitHub Actions.

Required env:
- OPENAI_API_KEY
- YOUTUBE_URL

Optional env:
- SERMON_TITLE
- SPEAKER
- BIBLE_TEXT
- OPENAI_TRANSCRIBE_MODEL, default: gpt-4o-transcribe
- OPENAI_TEXT_MODEL, default: gpt-5.5
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

ROOT = Path.cwd()
WORK_DIR = ROOT / "_work_sermon_archive"
AUDIO_DIR = WORK_DIR / "audio"
SERMONS_DIR = ROOT / "sermons"
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "sermons.json"

OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.5")

MAX_AUDIO_BYTES = 24 * 1024 * 1024
SEGMENT_SECONDS = 20 * 60


def log(message: str) -> None:
    print(f"[sermon-archive] {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    print(f"\n[sermon-archive:ERROR] {message}\n", file=sys.stderr, flush=True)
    sys.exit(exit_code)


def run(cmd: List[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        print(result.stdout)
        fail(f"명령 실행에 실패했습니다: {' '.join(cmd)}")
    return result


def ensure_env() -> Dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    youtube_url = os.getenv("YOUTUBE_URL", "").strip()

    if not api_key:
        fail("OPENAI_API_KEY가 없습니다. GitHub Secrets에 OPENAI_API_KEY를 등록했는지 확인하세요.")
    if not youtube_url:
        fail("YOUTUBE_URL이 없습니다. GitHub Actions 실행 시 유튜브 주소를 입력해야 합니다.")

    return {
        "api_key": api_key,
        "youtube_url": youtube_url,
        "sermon_title": os.getenv("SERMON_TITLE", "").strip(),
        "speaker": os.getenv("SPEAKER", "").strip(),
        "bible_text": os.getenv("BIBLE_TEXT", "").strip(),
    }


def reset_workdir() -> None:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SERMONS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_video_metadata(youtube_url: str) -> Dict[str, Any]:
    log("유튜브 영상 정보를 가져옵니다.")
    result = run(["yt-dlp", "--dump-json", "--no-playlist", youtube_url])
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        log("영상 정보를 JSON으로 읽지 못했습니다. 기본값으로 진행합니다.")
        data = {}

    upload_date = data.get("upload_date") or ""
    formatted_date = ""
    if re.fullmatch(r"\d{8}", upload_date):
        formatted_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

    return {
        "video_title": data.get("title") or "",
        "channel": data.get("uploader") or data.get("channel") or "",
        "upload_date": formatted_date,
        "webpage_url": data.get("webpage_url") or youtube_url,
        "duration": data.get("duration"),
    }


def download_audio(youtube_url: str) -> Path:
    log("유튜브 영상에서 오디오를 내려받습니다.")
    output_template = str(AUDIO_DIR / "download.%(ext)s")

    run([
        "yt-dlp",
        "--no-playlist",
        "-f", "bestaudio/best",
        "-o", output_template,
        youtube_url,
    ])

    candidates = [p for p in AUDIO_DIR.glob("download.*") if p.is_file()]
    if not candidates:
        fail("오디오 파일을 찾지 못했습니다. yt-dlp 다운로드 결과를 확인하세요.")

    audio_input = candidates[0]
    log(f"오디오 다운로드 완료: {audio_input.name}")
    return audio_input


def convert_to_small_mp3(audio_input: Path) -> Path:
    log("전사용 저용량 MP3로 변환합니다.")
    output = AUDIO_DIR / "full.mp3"

    run([
        "ffmpeg",
        "-y",
        "-i", str(audio_input),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "32k",
        str(output),
    ])

    size_mb = output.stat().st_size / (1024 * 1024)
    log(f"MP3 변환 완료: {size_mb:.2f}MB")
    return output


def split_audio(mp3_path: Path) -> List[Path]:
    size = mp3_path.stat().st_size
    if size <= MAX_AUDIO_BYTES:
        log("오디오가 25MB 이하이므로 분할하지 않습니다.")
        return [mp3_path]

    log("오디오가 커서 20분 단위로 분할합니다.")
    chunk_pattern = str(AUDIO_DIR / "chunk_%03d.mp3")
    run([
        "ffmpeg",
        "-y",
        "-i", str(mp3_path),
        "-f", "segment",
        "-segment_time", str(SEGMENT_SECONDS),
        "-c", "copy",
        chunk_pattern,
    ])

    chunks = sorted(AUDIO_DIR.glob("chunk_*.mp3"))
    if not chunks:
        fail("오디오 분할에 실패했습니다.")

    safe_chunks: List[Path] = []
    for chunk in chunks:
        if chunk.stat().st_size > MAX_AUDIO_BYTES:
            fail(f"{chunk.name} 파일이 여전히 25MB를 초과합니다. 비트레이트나 segment 시간을 줄여야 합니다.")
        safe_chunks.append(chunk)

    log(f"오디오 분할 완료: {len(safe_chunks)}개")
    return safe_chunks


def transcribe_audio(client: OpenAI, audio_chunks: List[Path]) -> str:
    log("OpenAI Speech-to-Text로 음성을 전사합니다.")
    transcripts: List[str] = []

    for idx, chunk in enumerate(audio_chunks, start=1):
        size_mb = chunk.stat().st_size / (1024 * 1024)
        log(f"전사 중: {chunk.name} ({idx}/{len(audio_chunks)}, {size_mb:.2f}MB)")

        with chunk.open("rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=OPENAI_TRANSCRIBE_MODEL,
                file=audio_file,
                response_format="text",
                language="ko",
                prompt="이 파일은 한국어 기독교 설교입니다. 성경 인명, 지명, 본문 표기를 가능한 한 정확하게 전사해 주세요.",
            )

        if isinstance(result, str):
            text = result
        elif hasattr(result, "text"):
            text = result.text
        else:
            text = str(result)

        transcripts.append(text.strip())

    raw_transcript = "\n\n".join(t for t in transcripts if t)
    if len(raw_transcript) < 100:
        fail("전사된 텍스트가 너무 짧습니다. 오디오 다운로드나 전사 과정을 확인하세요.")

    log(f"전사 완료: 약 {len(raw_transcript):,}자")
    return raw_transcript


def build_analysis_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "speaker": {"type": "string"},
            "date": {"type": "string"},
            "bibleText": {"type": "string"},
            "church": {"type": "string"},
            "summary": {"type": "string"},
            "mainMessage": {"type": "string"},
            "outline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["title", "summary"],
                },
            },
            "topics": {"type": "array", "items": {"type": "string"}},
            "applications": {"type": "array", "items": {"type": "string"}},
            "illustrations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "topics": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "summary", "topics"],
                },
            },
            "correctedTranscript": {"type": "string"},
        },
        "required": [
            "title",
            "speaker",
            "date",
            "bibleText",
            "church",
            "summary",
            "mainMessage",
            "outline",
            "topics",
            "applications",
            "illustrations",
            "correctedTranscript",
        ],
    }


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    fail("AI 응답을 JSON으로 읽지 못했습니다. 응답 형식을 확인해야 합니다.")
    return {}


def analyze_sermon(
    client: OpenAI,
    *,
    youtube_url: str,
    raw_transcript: str,
    video_meta: Dict[str, Any],
    user_title: str,
    user_speaker: str,
    user_bible_text: str,
) -> Dict[str, Any]:
    log("전사문을 오타 수정하고 설교 내용을 분석합니다.")

    prompt = f"""
다음은 유튜브 설교 영상을 음성 전사한 원문입니다.

목표:
1. correctedTranscript에는 원문 흐름을 보존하면서 오타, 띄어쓰기, 문장부호, 성경 인명과 지명 표기만 수정한 설교문 전문을 넣으십시오.
2. correctedTranscript에서 요약, 문장 재구성, 설교자 어투 변경, 내용 추가, 내용 삭제를 하지 마십시오.
3. 나머지 항목에는 설교 분석 결과를 넣으십시오.
4. 설교 안에 대지가 명확히 있으면 outline에 그 대지를 반영하십시오. 대지가 명확하지 않으면 자연스러운 논리 흐름에 따라 정리하십시오.
5. 예화는 설교자가 든 이야기, 사례, 비유, 경험담을 중심으로 추출하십시오.
6. topics는 나중에 색인으로 쓸 수 있도록 5~12개 정도로 정리하십시오.
7. 확인할 수 없는 정보는 "미상"으로 표기하십시오.

사용자가 입력한 보조 정보:
- 설교 제목: {user_title or "미입력"}
- 설교자: {user_speaker or "미입력"}
- 본문: {user_bible_text or "미입력"}

유튜브 메타데이터:
- 영상 제목: {video_meta.get("video_title") or "미상"}
- 채널: {video_meta.get("channel") or "미상"}
- 업로드일: {video_meta.get("upload_date") or "미상"}
- 주소: {youtube_url}

전사 원문:
{raw_transcript}
"""

    schema = build_analysis_schema()

    try:
        response = client.responses.create(
            model=OPENAI_TEXT_MODEL,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "sermon_archive_result",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        text = response.output_text
    except Exception as structured_error:
        log(f"구조화 출력 요청이 실패했습니다. 일반 JSON 프롬프트로 재시도합니다: {structured_error}")
        fallback_prompt = prompt + """

반드시 JSON 객체만 출력하십시오.
마크다운 코드블록을 쓰지 마십시오.
출력 JSON의 키는 다음과 같아야 합니다:
title, speaker, date, bibleText, church, summary, mainMessage, outline, topics, applications, illustrations, correctedTranscript
"""
        response = client.responses.create(
            model=OPENAI_TEXT_MODEL,
            input=fallback_prompt,
        )
        text = response.output_text

    analysis = parse_json_object(text)

    if user_title:
        analysis["title"] = user_title
    if user_speaker:
        analysis["speaker"] = user_speaker
    if user_bible_text:
        analysis["bibleText"] = user_bible_text

    if not analysis.get("title") or analysis.get("title") == "미상":
        analysis["title"] = video_meta.get("video_title") or "제목 미상"
    if not analysis.get("speaker") or analysis.get("speaker") == "미상":
        analysis["speaker"] = user_speaker or video_meta.get("channel") or "설교자 미상"
    if not analysis.get("date") or analysis.get("date") == "미상":
        analysis["date"] = video_meta.get("upload_date") or dt.date.today().isoformat()
    if not analysis.get("church") or analysis.get("church") == "미상":
        analysis["church"] = video_meta.get("channel") or "미상"

    for key in ["outline", "topics", "applications", "illustrations"]:
        if not isinstance(analysis.get(key), list):
            analysis[key] = []

    if not analysis.get("correctedTranscript"):
        analysis["correctedTranscript"] = raw_transcript

    return analysis


def safe_slug(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^0-9a-z가-힣]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:90] or "sermon"


def load_sermons_json() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        return []

    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("sermons"), list):
        return data["sermons"]
    return []


def unique_sermon_dir(base_slug: str, video_url: str) -> Path:
    current = load_sermons_json()
    for item in current:
        if item.get("videoUrl") == video_url and item.get("id"):
            return SERMONS_DIR / item["id"]

    candidate = SERMONS_DIR / base_slug
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        next_candidate = SERMONS_DIR / f"{base_slug}-{index}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_sermons_json(item: Dict[str, Any]) -> None:
    sermons = load_sermons_json()
    replaced = False

    for idx, old in enumerate(sermons):
        if old.get("id") == item.get("id") or old.get("videoUrl") == item.get("videoUrl"):
            sermons[idx] = item
            replaced = True
            break

    if not replaced:
        sermons.insert(0, item)

    write_json(DATA_FILE, {"sermons": sermons})


def make_sermon_card_markdown(item: Dict[str, Any]) -> str:
    outline = "\n".join(
        f"{idx + 1}. **{o.get('title', '대지 제목 미상')}** — {o.get('summary', '')}"
        for idx, o in enumerate(item.get("outline", []))
    )

    illustrations = "\n".join(
        f"- **{i.get('title', '예화 제목 미상')}**: {i.get('summary', '')} "
        f"(주제: {', '.join(i.get('topics', []))})"
        for i in item.get("illustrations", [])
    )

    topics = ", ".join(item.get("topics", []))
    applications = "\n".join(f"- {a}" for a in item.get("applications", []))

    return f"""# {item.get("title", "제목 미상")}

- 설교자: {item.get("speaker", "설교자 미상")}
- 날짜: {item.get("date", "날짜 미상")}
- 본문: {item.get("bibleText", "본문 미상")}
- 교회/채널: {item.get("church", "미상")}
- 영상: {item.get("videoUrl", "")}

## 핵심 메시지

{item.get("mainMessage", "")}

## 설교 요약

{item.get("summary", "")}

## 대지

{outline or "대지 정보가 없습니다."}

## 적용점

{applications or "적용점 정보가 없습니다."}

## 주제 색인

{topics or "주제 정보가 없습니다."}

## 예화

{illustrations or "추출된 예화가 없습니다."}
"""


def save_outputs(
    *,
    youtube_url: str,
    raw_transcript: str,
    analysis: Dict[str, Any],
    video_meta: Dict[str, Any],
) -> Dict[str, Any]:
    title = analysis.get("title", "제목 미상")
    date = analysis.get("date", dt.date.today().isoformat())
    base_slug = safe_slug(f"{date}-{title}")
    sermon_dir = unique_sermon_dir(base_slug, youtube_url)
    sermon_id = sermon_dir.name

    sermon_dir.mkdir(parents=True, exist_ok=True)

    raw_path = sermon_dir / "raw_transcript.md"
    corrected_path = sermon_dir / "corrected_transcript.md"
    analysis_path = sermon_dir / "analysis.json"
    illustrations_path = sermon_dir / "illustrations.json"
    metadata_path = sermon_dir / "metadata.json"
    card_path = sermon_dir / "sermon_card.md"

    corrected = analysis.get("correctedTranscript", raw_transcript)

    metadata = {
        "id": sermon_id,
        "videoUrl": youtube_url,
        "videoTitle": video_meta.get("video_title", ""),
        "channel": video_meta.get("channel", ""),
        "uploadDate": video_meta.get("upload_date", ""),
        "duration": video_meta.get("duration"),
        "createdAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "transcribeModel": OPENAI_TRANSCRIBE_MODEL,
        "textModel": OPENAI_TEXT_MODEL,
    }

    item = {
        "id": sermon_id,
        "videoUrl": youtube_url,
        "title": title,
        "speaker": analysis.get("speaker", "설교자 미상"),
        "date": date,
        "bibleText": analysis.get("bibleText", "본문 미상"),
        "church": analysis.get("church", video_meta.get("channel") or "미상"),
        "summary": analysis.get("summary", ""),
        "mainMessage": analysis.get("mainMessage", ""),
        "outline": analysis.get("outline", []),
        "topics": analysis.get("topics", []),
        "applications": analysis.get("applications", []),
        "illustrations": analysis.get("illustrations", []),
        "createdAt": metadata["createdAt"],
        "files": {
            "rawTranscript": str(raw_path).replace("\\", "/"),
            "correctedTranscript": str(corrected_path).replace("\\", "/"),
            "analysis": str(analysis_path).replace("\\", "/"),
            "illustrations": str(illustrations_path).replace("\\", "/"),
            "metadata": str(metadata_path).replace("\\", "/"),
            "sermonCard": str(card_path).replace("\\", "/"),
        },
    }

    analysis_for_file = dict(analysis)
    analysis_for_file.pop("correctedTranscript", None)

    write_text(raw_path, raw_transcript)
    write_text(corrected_path, corrected)
    write_json(analysis_path, analysis_for_file)
    write_json(illustrations_path, analysis.get("illustrations", []))
    write_json(metadata_path, metadata)
    write_text(card_path, make_sermon_card_markdown(item))

    update_sermons_json(item)

    log(f"결과 저장 완료: {sermon_dir}")
    return item


def main() -> None:
    env = ensure_env()
    reset_workdir()

    client = OpenAI(api_key=env["api_key"])

    video_meta = get_video_metadata(env["youtube_url"])
    downloaded_audio = download_audio(env["youtube_url"])
    mp3 = convert_to_small_mp3(downloaded_audio)
    chunks = split_audio(mp3)
    raw_transcript = transcribe_audio(client, chunks)

    analysis = analyze_sermon(
        client,
        youtube_url=env["youtube_url"],
        raw_transcript=raw_transcript,
        video_meta=video_meta,
        user_title=env["sermon_title"],
        user_speaker=env["speaker"],
        user_bible_text=env["bible_text"],
    )

    item = save_outputs(
        youtube_url=env["youtube_url"],
        raw_transcript=raw_transcript,
        analysis=analysis,
        video_meta=video_meta,
    )

    log("완료되었습니다.")
    log(f"설교 ID: {item['id']}")
    log(f"제목: {item['title']}")


if __name__ == "__main__":
    main()
