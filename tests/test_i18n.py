"""ADR-033: Turkish and English for everything a person reads."""
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from cadai import i18n, jobs
from cadai.i18n import say

STATIC = Path(__file__).resolve().parents[1] / "cadai" / "web" / "static"


def test_a_message_reads_in_the_current_language_and_keeps_both():
    m = say("Kural yok", "No rules")
    assert str(m) == "Kural yok" and f"{m}" == "Kural yok"
    token = i18n.use("en")
    try:
        assert str(m) == "No rules"
    finally:
        i18n.reset(token)
    assert m.both() == {"tr": "Kural yok", "en": "No rules"}


def test_job_lines_keep_both_languages_and_old_lines_still_read():
    job = jobs.Job(id="t", kind="t", title={"tr": "İş", "en": "Job"}, params={})
    job.log(say("3 oda", "3 rooms"))
    job.log("DXF 12 MB")                                     # reads the same in both
    lines = job.summary()["lines"]
    assert lines[0]["tr"] == "3 oda" and lines[0]["en"] == "3 rooms"
    assert lines[1]["tr"] == lines[1]["en"] == "DXF 12 MB"
    old = jobs.Job(id="o", kind="o", title="Eski iş", params={}, lines=["12:00:00 eski satır"])
    s = old.summary()
    assert s["title"]["tr"] == "Eski iş" and s["lines"][0] == {"t": "12:00:00", "tr": "eski satır", "en": "eski satır"}


def test_stored_turkish_text_gets_english_from_the_memory(tmp_path, monkeypatch):
    memory = tmp_path / "translations.json"
    memory.write_text(json.dumps({"kesit": "section"}), encoding="utf-8")
    monkeypatch.setattr(i18n, "MEMORY_FILE", memory)
    monkeypatch.setitem(i18n._MEMORY, "mtime", None)
    assert i18n.bi("kesit") == {"tr": "kesit", "en": "section"}
    assert i18n.bi("bilinmeyen") == {"tr": "bilinmeyen", "en": "bilinmeyen"}
    assert i18n.bi({"tr": "a", "en": "b"}) == {"tr": "a", "en": "b"}


def test_the_switch_cookie_sets_the_language_of_server_texts():
    from cadai.web.app import app
    c = TestClient(app)
    assert c.post("/api/jobs/nope", json={}).json()["detail"] == "bilinmeyen iş: nope"
    c.cookies.set(i18n.COOKIE, "en")
    assert c.post("/api/jobs/nope", json={}).json()["detail"] == "unknown job: nope"
    labels = c.get("/api/layer-types").json()["labels"]
    assert labels["wall"] == {"tr": "Duvar", "en": "Wall"}


def test_download_names_follow_the_language(tmp_path):
    from cadai.web.app import download_name
    d = tmp_path / "out"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"source": r"C:\x\Plan A.dwg", "mode": "agent"}), encoding="utf-8")
    assert download_name(d / "clean.dwg") == "Plan A - Claude temizliği.dwg"
    token = i18n.use("en")
    try:
        assert download_name(d / "clean.dwg") == "Plan A - Claude cleanup.dwg"
        assert download_name(d / "placed_corrected.dwg") == "Plan A - electrical layout (corrected).dwg"
    finally:
        i18n.reset(token)


TURKISH = re.compile(r"[çğıöşüÇĞİÖŞÜ]")


def _without_t_calls(code: str) -> str:
    """The code with every T(...) call removed (brackets counted outside strings)."""
    out, i = [], 0
    for m in re.finditer(r"(?<![\w.])T\(", code):
        if m.start() < i:
            continue
        out.append(code[i:m.start()])
        depth, j, quote = 0, m.end() - 1, None
        while j < len(code):
            ch = code[j]
            if quote:
                if ch == "\\":
                    j += 1
                elif ch == quote:
                    quote = None
            elif ch in "\"'`":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        i = j + 1
    out.append(code[i:])
    return "".join(out)


def test_every_turkish_page_string_has_its_english():
    """Turkish in page code only inside T("tr", "en"); in markup only with a data-en twin."""
    for page in list(STATIC.glob("*.html")) + [STATIC / "common.js"]:
        text = page.read_text(encoding="utf-8")
        code = text if page.suffix == ".js" else "\n".join(re.findall(r"<script>(.*?)</script>", text, re.S))
        for line in _without_t_calls(code).splitlines():
            if not line.strip().startswith("//") and "in its own language" not in line:
                assert not TURKISH.search(line), f"{page.name}: {line.strip()[:90]!r} outside T()"
        if page.suffix == ".js":
            continue
        markup = re.sub(r"<script>.*?</script>|<style>.*?</style>", "", text, flags=re.S)
        for m in re.finditer(r"<(\w+)([^>]*)>([^<]*)", markup):
            tag, attrs, inner = m.groups()
            if TURKISH.search(inner) and "data-en" not in attrs:
                # text after an inline tag belongs to the enclosing element's data-en-html
                before = markup[:m.start()].rsplit("data-en-html", 1)
                assert len(before) == 2 and "</div>" not in before[1] and "</p>" not in before[1], \
                    f"{page.name}: <{tag}> {inner.strip()[:60]!r} has no English"


def test_pages_and_scripts_are_revalidated_so_a_switch_is_never_stale():
    from cadai.web.app import app
    c = TestClient(app)
    for path in ("/demo", "/draw", "/static/i18n.js", "/static/common.js"):
        assert c.get(path).headers.get("cache-control") == "no-cache", path
