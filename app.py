from flask import Flask, request, render_template, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
import uuid
import shutil
import zipfile
from pathlib import Path

# Import your existing code (UNCHANGED)
from create_rubric import ResumeRubricGenerator
from score_resumes import ResumeScorer


ALLOWED_EXTS = {".pdf", ".docx", ".txt"}

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"  # each request gets a unique run folder
RUNS_DIR.mkdir(exist_ok=True)

#app = Flask(__name__)
app = Flask(__name__, template_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB total upload cap


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTS


def zip_folder(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(src_dir)))


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process():
    # ----- inputs -----
    anthropic_key = (request.form.get("anthropic_key") or "").strip()
    google_key = (request.form.get("google_key") or "").strip()
    prompt = (request.form.get("prompt") or "").strip()
    ensemble = (request.form.get("ensemble") or "true").lower() != "false"

    if not prompt:
        return jsonify({"error": "Missing required field: prompt"}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded"}), 400

    # ----- create run workspace -----
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    resumes_dir = run_dir / "resumes"
    out_dir = run_dir / "outputs"
    detailed_dir = run_dir / "rubric_scores"
    resumes_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- save resumes -----
    saved = []
    for f in files:
        if not f or not f.filename:
            continue
        filename = secure_filename(f.filename)
        if not allowed_file(filename):
            continue
        dst = resumes_dir / filename
        f.save(dst)
        saved.append(filename)

    if not saved:
        shutil.rmtree(run_dir, ignore_errors=True)
        return jsonify({"error": "No supported files. Upload .pdf, .docx, or .txt"}), 400

    # ----- set keys for THIS request only -----
    # Your scripts read env vars at __init__ time. We'll set them before we instantiate.
    # (We avoid printing/logging keys.)
    old_anth = os.environ.get("ANTHROPIC_API_KEY")
    old_goog = os.environ.get("GOOGLE_API_KEY")
    try:
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key

        # 1) Generate rubric -> rubric.json
        rubric_path = run_dir / "rubric.json"
        gen = ResumeRubricGenerator(resume_dir=str(resumes_dir))
        gen.load_resumes()
        rubric = gen.generate_rubric(prompt)
        gen.save_rubric(str(rubric_path))

        # 2) Score resumes -> rankings.xlsx + per-candidate JSONs
        scorer = ResumeScorer(resume_dir=str(resumes_dir), rubric_path=str(rubric_path))
        scorer.load_rubric()
        scorer.load_resumes()
        scorer.score_all_resumes(use_ensemble=ensemble)
        scorer.save_detailed_scores(output_dir=str(detailed_dir))

        rankings_xlsx = run_dir / "candidate_rankings.xlsx"
        scorer.create_summary_spreadsheet(output_file=str(rankings_xlsx))

        ranked = scorer.rank_candidates()

        # Zip detailed scores for download
        scores_zip = run_dir / "rubric_scores.zip"
        zip_folder(detailed_dir, scores_zip)

        # Prepare response (lightweight + useful)
        response = {
            "run_id": run_id,
            "rubric_download": f"/downloads/{run_id}/rubric.json",
            "rankings_download": f"/downloads/{run_id}/candidate_rankings.xlsx",
            "detailed_scores_download": f"/downloads/{run_id}/rubric_scores.zip",
            "results": [
                {
                    "filename": r["filename"],
                    "composite_score": r.get("composite_score"),
                    "total_crackedness": r.get("total_crackedness"),
                    "total_fit": r.get("total_fit"),
                    "candidate_description": r.get("candidate_description"),
                    "strengths_explanation": r.get("strengths_explanation"),
                    "rank": r.get("rank"),
                }
                for r in ranked
            ],
        }
        return jsonify(response)

    except Exception as e:
        # Clean up failed run to avoid clutter
        shutil.rmtree(run_dir, ignore_errors=True)
        return jsonify({"error": str(e)}), 500

    finally:
        # restore env
        if old_anth is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old_anth

        if old_goog is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = old_goog


@app.route("/downloads/<run_id>/<path:filename>", methods=["GET"])
def downloads(run_id, filename):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        abort(404)

    # Only allow serving files inside the run directory
    target = (run_dir / filename).resolve()
    if run_dir.resolve() not in target.parents and target != run_dir.resolve():
        abort(403)

    if not target.exists() or not target.is_file():
        abort(404)

    return send_from_directory(directory=str(run_dir), path=filename, as_attachment=True)


if __name__ == "__main__":
    # For local dev only. For production, run via gunicorn/uvicorn.
    app.run(host="0.0.0.0", port=5001, debug=True)
