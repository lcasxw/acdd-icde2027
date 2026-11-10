"""Restore the six v50 vectors with the manuscript's three figure-label amendments.

Run with pypdf and provide the original Matplotlib 3.11.1 interpreter for Fig. 7.
The input tree is read-only. All intermediate and original-renderer outputs go to work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, FloatObject, NameObject, TextStringObject

HERE = Path(__file__).resolve().parents[1]
ORIGINALS = {
    1: "figure1_acdd_workflow", 2: "figure2_construct_protected_audit",
    3: "figure2_cross_cohort_forest", 4: "figure3_access_matched_controls",
    5: "figure4_source_robustness_radar", 6: "figure6_oracle_sensitivity",
    7: "figure7_participant_degradation",
}
WORKFLOW = {"Access-conditioned deployment audit": "Access-conditioned decision audit",
            "No target labels in fit": "At the score layer"}
PARTICIPANTS = {"n=62/model": "15-model means, n=62", "Each model, >2 pp": "Per backbone, >2 pp"}
INTERVAL_FOOTER = {
    "Gain CIs are frozen; other CIs are descriptive. Oracle headroom is a separate reference.":
    "Gain CIs use the post-confirmation bootstrap. Other CIs are descriptive. Oracle headroom is a separate reference."
}
F1 = "figure1_revision_v49_20261014"
F7 = "figure7_single_column_approved_20261008"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def substitute(text, changes):
    for old, new in changes.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def annotate_workflow(source, target):
    writer = PdfWriter(clone_from=source)
    page = writer.pages[0]
    content = page.get_contents()
    ops = content.operations
    seen = set()
    for i, (args, op) in enumerate(ops):
        if op != b"Tj" or str(args[0]) not in WORKFLOW:
            continue
        old = str(args[0])
        new = WORKFLOW[old]
        transform, font = ops[i-3], ops[i-2]
        assert transform[1] == b"Tm" and font[1] == b"Tf" and old not in seen
        # The title is left aligned. Retain the center of the original blue-card label.
        if old == "No target labels in fit":
            name, size = font[0]
            face = page["/Resources"]["/Font"][name]
            widths, first = face["/Widths"], int(face["/FirstChar"])
            def width(s):
                return sum(float(widths[ord(c)-first]) for c in s) * float(size) / 1000
            transform[0][4] = FloatObject(float(transform[0][4]) + (width(old)-width(new))/2)
        args[0] = TextStringObject(new)
        seen.add(old)
    assert seen == set(WORKFLOW)
    content.operations = ops
    serialized = DecodedStreamObject()
    serialized.set_data(content.get_data())
    page[NameObject("/Contents")] = writer._add_object(serialized.flate_encode())
    writer.write(target)


def annotate_intervals(source, target):
    """Change one Identity-H text operand, preserving every vector drawing operator."""
    writer = PdfWriter(clone_from=source)
    page = writer.pages[0]
    original = page.get_contents().get_data()
    old, new = next(iter(INTERVAL_FOOTER.items()))
    old_bytes, new_bytes = old.encode("utf-16-be"), new.encode("utf-16-be")
    assert page["/Resources"]["/Font"]["/F1"]["/Encoding"] == "/Identity-H"
    assert original.count(old_bytes) == 1
    stream = DecodedStreamObject()
    stream.set_data(original.replace(old_bytes, new_bytes))
    page[NameObject("/Contents")] = writer._add_object(stream.flate_encode())
    writer.write(target)
    placed = PdfReader(target).pages[0]
    assert new in placed.extract_text() and old not in placed.extract_text()
    assert placed.get_contents().get_data().replace(new_bytes, old_bytes) == original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-paper", type=Path, default=HERE.parent/"tnsre_acdd")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--plot-python", type=Path, required=True)
    args = parser.parse_args()
    paper, out = args.input_paper.resolve(), args.out.resolve()
    assert out != paper and paper not in out.parents
    dest = out/"figures"
    dest.mkdir(parents=True, exist_ok=True)
    (out/"work").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="figure-labels-", dir=out/"work"))
    src = paper/"figures"
    for n in (3, 5, 6):
        filename = ORIGINALS[n]+".pdf"
        shutil.copyfile(src/filename, dest/filename)
    filename = ORIGINALS[2]+".pdf"
    annotate_intervals(src/filename, dest/filename)
    filename = ORIGINALS[1]+".pdf"
    annotate_workflow(src/filename, dest/filename)
    vector = paper/"editable_figures"/F1/"figure1_acdd_workflow_v49.svg"
    (work/"workflow.svg").write_text(substitute(vector.read_text(encoding="utf-8"), WORKFLOW), encoding="utf-8")
    native = next(vector.parent.glob("*_native_spec.json"))
    (work/"workflow_native_spec.json").write_text(substitute(native.read_text(encoding="utf-8"), WORKFLOW), encoding="utf-8")

    approved = paper/"editable_figures"/F7
    mirror = work/"paper/tnsre_acdd"
    amended = mirror/"editable_figures"/F7
    amended.mkdir(parents=True)
    (mirror/"tools").mkdir()
    shutil.copyfile(paper/"tools/plot_style.py", mirror/"tools/plot_style.py")
    shutil.copyfile(approved/"ccfa_paper.mplstyle", amended/"ccfa_paper.mplstyle")
    code = substitute((approved/"build_figure.py").read_text(encoding="utf-8"), PARTICIPANTS)
    script = amended/"build_figure.py"
    script.write_text(code, encoding="utf-8")
    output = work/"participant"
    command = [str(args.plot_python.resolve()), "-B", str(script), "--source", str(paper/"source_data"), "--output", str(output)]
    run = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace",
                         env=dict(os.environ, MPLCONFIGDIR=str(work/"mpl"), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1"))
    (work/"renderer.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    if run.returncode:
        raise RuntimeError(run.stderr)
    baseline_producer = PdfReader(src/(ORIGINALS[7]+".pdf")).metadata.producer
    assert PdfReader(output/"figure7_single_column.pdf").metadata.producer == baseline_producer, baseline_producer
    shutil.copyfile(output/"figure7_single_column.pdf", dest/(ORIGINALS[7]+".pdf"))
    editable = out/"editable"
    editable.mkdir(exist_ok=True)
    for name in ("workflow.svg","workflow_native_spec.json"):
        shutil.copyfile(work/name, editable/name)
    shutil.copyfile(output/"figure7_single_column.svg", editable/"participant_means.svg")
    for n, labels in ((1,WORKFLOW), (7,PARTICIPANTS)):
        actual = PdfReader(dest/(ORIGINALS[n]+".pdf")).pages[0].extract_text()
        assert all(old not in actual and new in actual for old,new in labels.items()), actual
    log = out/"logs"
    log.mkdir(exist_ok=True)
    record = {"status":"PASS", "work":str(work), "renderer":command, "fig1_labels":WORKFLOW, "fig2_labels":INTERVAL_FOOTER, "fig7_labels":PARTICIPANTS,
              "originals":{str(n):sha(src/(stem+".pdf")) for n,stem in ORIGINALS.items()},
              "included":{str(n):sha(dest/(ORIGINALS[n]+".pdf")) for n in (1,2,3,5,6,7)}}
    (log/"figure_assets.json").write_text(json.dumps(record,indent=2)+"\n", encoding="utf-8")
    print("Placed v50 figures 1, 2, 3, 5, 6, 7.", work)


if __name__ == "__main__":
    main()
