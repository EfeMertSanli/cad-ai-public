"""ADR-032: drawings leave the office only without personal details (names may stay)."""
import ezdxf

from cadai.demo.anonymise import anonymise, is_private, leaks


def test_personal_details_are_recognised_and_drawing_text_is_not():
    for text in ("ADRES: Örnek Mah. Deneme Cad. No:1/2", "Tel: 0500 000 0000", "\A1;Tel:0(212) 000 00 00",
                 "E-Mail: someone.59@gmail.com", "Web: www.example.com", "PARSEL NO.", "ADA NO.", "ÖRNEK SOKAK",
                 "Deneme Sok.", "BEL.SİCİL NO", "BÜRO TES.NO", "Örnek Plaza 1/101"):
        assert is_private(text), text
    for text in ("Salon", "E.Yatak Odası", "12.35 m²", "Tzk-0.88(166.07)", "Mahal Adı", "Cevriye BERKOĞLU",
                 "Dbl Plain - 120-150-2171602-ZEMİN KAT", "1.4.4 @ 2026-10-21T21:14:21"):
        assert not is_private(text), text


def test_outside_the_plans_no_text_survives_inside_only_personal_text_goes(tmp_path):
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_text("Salon", dxfattribs={"insert": (5, 5)})
    msp.add_text("ADRES: Deneme Cad. No:1", dxfattribs={"insert": (5, 6)})
    msp.add_text("1182", dxfattribs={"insert": (50, 50)})                       # a parcel number: a bare value
    title = doc.blocks.new("ANTET")
    title.add_text("Tel: 0500 000 0000")
    title.add_text("Mimar")
    door = doc.blocks.new("DOOR")
    door.add_text("D1")
    msp.add_blockref("ANTET", (60, 60))
    msp.add_blockref("DOOR", (2, 2))
    src, out = tmp_path / "raw.dxf", tmp_path / "anon.dxf"
    doc.saveas(src)
    assert leaks(src)                                                          # the scan sees the original
    report = anonymise(src, out, [(0, 0, 10, 10)], margin=0.5)
    kept = ezdxf.readfile(out)
    texts = sorted(e.dxf.text for e in kept.modelspace() if e.dxftype() == "TEXT")
    assert texts == ["Salon"]
    assert [e.dxf.text for e in kept.blocks["ANTET"] if e.dxftype() == "TEXT"] == []
    assert [e.dxf.text for e in kept.blocks["DOOR"] if e.dxftype() == "TEXT"] == ["D1"]
    assert report["outside"] == 1 and report["private"] == 1
    assert leaks(out) == []
