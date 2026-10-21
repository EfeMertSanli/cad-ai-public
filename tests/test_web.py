from fastapi.testclient import TestClient


def test_pages_and_api_respond():
    from cadai.web.app import app
    c = TestClient(app)
    for page in ("/", "/clean", "/lexicon/symbols", "/lexicon/furniture", "/debug", "/project/x", "/static/common.js"):
        assert c.get(page).status_code == 200, page
    ov = c.get("/api/overview").json()
    assert [s["key"] for s in ov["steps"]] == ["inventory", "convert", "label"]
    assert c.get("/api/debug/system").json()["paths"]["data"]
    assert c.get("/api/outputs").json() == []
    assert c.post("/api/jobs/nope", json={}).status_code == 404
    bad = c.post("/api/clean/upload", files={"file": ("x.txt", b"hi")})
    assert bad.status_code == 400


def test_symbol_types_are_validated():
    from cadai.placement import symbols
    from cadai.web.app import app
    c = TestClient(app)
    assert c.get("/api/lexicon/symbols").json()["catalog"] == []
    assert c.post("/api/lexicon/symbols/type", json={"key": "A", "type": "socket"}).status_code == 200
    assert c.post("/api/lexicon/symbols/type", json={"key": "A", "type": "bogus"}).status_code == 400
    assert c.get("/api/lexicon/nope").status_code == 404
    assert symbols.load_types() == {"A": "socket"}


def test_password_protects_everything(monkeypatch):
    from cadai import config
    from cadai.web.app import app
    monkeypatch.setattr(config, "PASSWORD", "gizli")
    c = TestClient(app)
    assert c.get("/").status_code == 401
    assert c.get("/api/overview", auth=("muhendis", "yanlis")).status_code == 401
    assert c.get("/api/overview", auth=("muhendis", "gizli")).status_code == 200


def test_hint_proposals():
    from cadai.placement.symbols import hint_type, propose_from_hints
    assert hint_type("J 60 W") == "light_ceiling" and hint_type("P 2X40 W") == "light_ceiling"
    assert hint_type("TOPRAKLI PRIZ") == "socket" and hint_type("TRIFAZE PRIZ") == "socket_power"
    assert hint_type("KOMUTATOR ANAHTAR") == "switch" and hint_type("315") is None
    p = propose_from_hints([{"block": "A", "hints": [["J 40 W", 8], ["1", 2]]}])
    assert p["A"]["type"] == "light_ceiling" and p["A"]["confidence"] == {"tr": "yüksek", "en": "high"}


def test_download_names_say_which_cleanup(tmp_path):
    import json as _json
    from cadai.web.app import download_name
    for engine, expected in (("legacy", "eski sistem temizliği"), ("model", "model temizliği")):
        d = tmp_path / engine
        d.mkdir()
        (d / "summary.json").write_text(_json.dumps({"source": r"C:\x\100 1 Örnek.dwg", "engine": engine}),
                                        encoding="utf-8")
        assert download_name(d / "clean.dwg") == f"100 1 Örnek - {expected}.dwg"


def test_outputs_newest_first(tmp_path, monkeypatch):
    import json as _json
    import os
    from cadai.web import app as web
    monkeypatch.setattr(web, "OUTPUTS", tmp_path)
    for name, t in (("100-2-UYGULAMA-20261010-184052", 2000), ("100 Örnek-20261010-183512", 1000)):
        (tmp_path / name).mkdir()
        s = tmp_path / name / "summary.json"
        s.write_text(_json.dumps({"source": name}), encoding="utf-8")
        os.utime(s, (t, t))
    names = [o["name"] for o in TestClient(web.app).get("/api/outputs").json()]
    assert names == ["100-2-UYGULAMA-20261010-184052", "100 Örnek-20261010-183512"]


def test_engineer_notes_on_lexicon_entries():
    from cadai.web.app import app
    c = TestClient(app)
    assert c.get("/api/lexicon/symbols").json()["notes"] == {}
    r = c.post("/api/lexicon/symbols/note", json={"key": "PRIZ1", "text": " Tezgah üstü etanj priz "})
    assert r.status_code == 200
    assert c.get("/api/lexicon/symbols").json()["notes"]["PRIZ1"]["text"] == "Tezgah üstü etanj priz"
    assert c.get("/api/lexicon/furniture").json()["notes"] == {}             # kept per lexicon
    c.post("/api/lexicon/symbols/note", json={"key": "PRIZ1", "text": ""})   # empty text removes it
    assert c.get("/api/lexicon/symbols").json()["notes"] == {}
    assert c.post("/api/lexicon/nope/note", json={"key": "A", "text": "x"}).status_code == 404


def test_relayer_saves_layer_types_and_cleans_again(tmp_path, monkeypatch):
    import json as _json
    from cadai.cleanup import layers
    from cadai.web import app as web
    monkeypatch.setattr(web, "OUTPUTS", tmp_path)
    submitted = []

    class Job:
        def summary(self):
            return {"id": "j1"}
    monkeypatch.setattr(web.runner, "submit", lambda kind, **kw: submitted.append((kind, kw)) or Job())
    (tmp_path / "out1").mkdir()
    (tmp_path / "out1" / "summary.json").write_text(_json.dumps({"source": r"C:\x\a.dwg", "keep_furniture": False,
                                                                 "mode": "layers", "engine": "model"}), encoding="utf-8")
    c = TestClient(web.app)
    r = c.post("/api/outputs/out1/relayer", json={"types": {"sketch_Pen_No__101": "wall"}, "mode": "auto"})
    assert r.status_code == 200 and submitted == [("clean", {"path": r"C:\x\a.dwg", "keep_furniture": False, "mode": "auto"})]
    assert layers.confirmed() == {"SKETCH_PEN_NO__101": "wall"}
    assert c.post("/api/outputs/out1/relayer", json={"types": {"A": "bogus"}}).status_code == 400
    assert c.post("/api/outputs/nope/relayer", json={"types": {}}).status_code == 404
    assert "wall" in c.get("/api/layer-types").json()["labels"]
    assert web.download_name(tmp_path / "out1" / "clean.dwg") == "a - katman temizliği.dwg"


def test_place_sends_confirmed_rules_and_rule_book_round_trips(tmp_path, monkeypatch):
    from cadai.web import app as web
    monkeypatch.setattr(web, "OUTPUTS", tmp_path)
    submitted = []

    class Job:
        def summary(self):
            return {"id": "j1"}
    monkeypatch.setattr(web.runner, "submit", lambda kind, **kw: submitted.append((kind, kw)) or Job())
    (tmp_path / "out1").mkdir()
    (tmp_path / "out1" / "clean.dxf").write_text("", encoding="utf-8")
    c = TestClient(web.app)
    rule = {"class": "socket", "room_types": ["living"], "min": 4, "max": None, "near": None}
    assert c.post("/api/place", json={"output": "out1", "instruction": "salonlara 4 priz", "rules": [rule]}).status_code == 200
    assert submitted[-1] == ("place", {"output": "out1", "instruction": "salonlara 4 priz", "rules": [rule]})
    c.post("/api/place", json={"output": "out1"})
    assert submitted[-1][1]["rules"] is None                  # no list: the job uses the rule book and translation
    book = c.get("/api/rulebook").json()
    assert book["rules"] == [] and "socket" in book["classes"] and "latch" in book["tags"]
    assert c.put("/api/rulebook", json={"rules": [rule, {"class": "x"}]}).json() == [rule]
    assert c.get("/api/rulebook").json()["rules"] == [rule]


def test_every_page_loads_the_english_layer_before_its_text():
    """ADR-031: the EN/TR switch translates in the browser, so every page loads i18n.js in <head>."""
    from cadai.web.app import app
    c = TestClient(app)
    for page in ("/", "/clean", "/draw", "/edit", "/lexicon/symbols", "/debug", "/project/x", "/demo"):
        html = c.get(page).text
        assert '<script src="/static/i18n.js"></script>' in html.split("</head>")[0], page
    assert c.get("/static/i18n.js").status_code == 200


def test_demo_files_stay_inside_the_demo_folder():
    from cadai.web.app import app
    c = TestClient(app)
    assert c.get("/demo-files/x/..%2F..%2Fcadai%2Fconfig.py").status_code == 404
    assert c.get("/demo-files/nope/facts.json").status_code == 404
    assert isinstance(c.get("/api/demo").json(), list)
