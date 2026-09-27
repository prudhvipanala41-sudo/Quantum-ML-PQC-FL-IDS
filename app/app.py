from flask import Flask, jsonify, render_template
from pathlib import Path
import json


app = Flask(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "final_experiments"
    / "final_model_comparison.json"
)

BENCHMARK_RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "benchmarking"
    / "benchmark_summary.json"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/api/results")
def results():

    if not FINAL_RESULTS_FILE.exists():
        return jsonify({
            "status": "error",
            "message": "Final experiment results file not found"
        }), 404

    with open(
        FINAL_RESULTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    return jsonify(data)

@app.route("/api/benchmark")
def benchmark():
    if not BENCHMARK_RESULTS_FILE.exists():
        return jsonify({
            "status": "error",
            "message": "Benchmark summary file not found"
        }), 404

    with open(BENCHMARK_RESULTS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return jsonify(data)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )