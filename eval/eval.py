"""Runner del golden set para eia-rag.

Uso (dev-only — corre contra una instancia LOCAL de eia-rag):

    uvicorn app.main:app --port 8000      # o: uv run dev
    uv run python eval/eval.py --endpoint http://localhost:8000/chat
    uv run python eval/eval.py --dry-run  # simula respuestas, no usa red

Ver eval/README.md para el detalle de formato y criterios.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from eval.naturalness import score_naturalness
except ImportError:  # ejecutado como script: sys.path[0] == eval/
    from naturalness import score_naturalness

from groq import AsyncGroq

try:
    from app.config import settings

    _DEFAULT_JUDGE_MODEL = getattr(settings, "GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
    _DEFAULT_JUDGE_KEY = getattr(settings, "GROQ_API_KEY", "")
except Exception:  # noqa: BLE001 — sin .env, el juez queda deshabilitado
    _DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
    _DEFAULT_JUDGE_KEY = ""

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET = EVAL_DIR / "golden_set.jsonl"
REPORT = EVAL_DIR / "report.json"

REGEX_PREFIX = "regex:"


def load_cases(path: Path) -> list[dict]:
    """Carga el golden set JSONL (una línea por caso)."""
    if not path.exists():
        raise SystemExit(f"No existe el dataset: {path}")
    cases: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSON inválido en {path.name}:{line_no}: {exc}") from exc
            cases.append(case)
    return cases


def criterion_matches(criterion: str, answer: str) -> bool:
    """Case-insensitive substring; con prefijo 'regex:' usa patrón."""
    if criterion.startswith(REGEX_PREFIX):
        pattern = criterion[len(REGEX_PREFIX):]
        try:
            return re.search(pattern, answer, re.IGNORECASE) is not None
        except re.error:
            return False
    return criterion.lower() in answer.lower()


def evaluate_case(case: dict, answers: list[str], intents: list[str] | None = None,
                  check_intent: bool = False) -> dict:
    """Un caso PASA si TODAS las 'debe_incluir' aparecen y NINGUNA
    'no_debe' aparece en las respuestas de todos sus turnos."""
    intents = intents or []
    checks: list[dict] = []
    ok = True

    for criterion in case.get("debe_incluir", []):
        passed = any(criterion_matches(criterion, a) for a in answers)
        checks.append({"tipo": "debe", "criterio": criterion, "ok": passed})
        ok = ok and passed

    for criterion in case.get("no_debe", []):
        passed = not any(criterion_matches(criterion, a) for a in answers)
        checks.append({"tipo": "no_debe", "criterio": criterion, "ok": passed})
        ok = ok and passed

    if check_intent and intents and case.get("intencion_esperada"):
        passed = any(i == case["intencion_esperada"] for i in intents)
        checks.append({"tipo": "intencion", "criterio": case["intencion_esperada"], "ok": passed})
        ok = ok and passed

    return {
        "id": case.get("id"),
        "tienda_id": case.get("tienda_id"),
        "inbox_id": case.get("inbox_id"),
        "tags": case.get("tags", []),
        "intencion_esperada": case.get("intencion_esperada"),
        "intencion_detectada": intents[-1] if intents else None,
        "turnos": len(case.get("turnos", [])),
        "respuestas_contenido": sum(1 for a in answers if a.strip()),
        "ok": ok,
        "checks": checks,
    }


def http_post_json(url: str, payload: dict, timeout: float) -> dict:
    """POST sincrónico a /chat (usado vía asyncio.to_thread). Nunca lanza."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"answer": "", "error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001 — fallback fail-safe para eval
        return {"answer": "", "error": str(exc)}


def build_payload(case: dict, index: int, user_base: int) -> dict:
    return {
        "query": case["turnos"][index],
        "conversation_id": f"eval-{case['id']}",
        "inbox_id": case["inbox_id"],
        "user_id": user_base + index,
        "channel": case.get("canal", "whatsapp"),
    }


def fake_answers(case: dict) -> tuple[list[str], list[str]]:
    """Respuestas simuladas para --dry-run: concatena los 'debe_incluir'."""
    base = " ".join(case.get("debe_incluir", []))
    return [base] * len(case.get("turnos", [])), []


async def run_case(case: dict, endpoint: str, timeout: float, check_intent: bool,
                   dry_run: bool, judge: Any | None = None,
                   judge_model: str = "") -> dict:
    if not case.get("turnos"):
        return evaluate_case(case, [], [], check_intent)

    if dry_run:
        answers, intents = fake_answers(case)
        return evaluate_case(case, answers, intents, check_intent)

    answers: list[str] = []
    intents: list[str] = []
    user_base = 900000 + hash(case.get("id", "")) % 100000
    for index in range(len(case["turnos"])):
        payload = build_payload(case, index, user_base)
        resp = await asyncio.to_thread(http_post_json, endpoint, payload, timeout)
        answers.append(resp.get("answer", ""))
        intents.append(resp.get("intent_detected", ""))
        if resp.get("error"):
            break

    result = evaluate_case(case, answers, intents, check_intent)

    if judge is not None and answers:
        scores = []
        for query, answer in zip(case["turnos"], answers):
            score = await score_naturalness(query, answer, judge, judge_model)
            scores.append(score)
        promedio = round(
            sum(s["puntaje"] for s in scores) / len(scores), 2
        ) if scores else 0
        result["naturalidad"] = {"por_turno": scores, "promedio": promedio}

    return result


def summarize(results: list[dict], args: argparse.Namespace) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["ok"])

    by_tag: dict[str, dict] = {}
    for r in results:
        for tag in r.get("tags", []):
            bucket = by_tag.setdefault(tag, {"total": 0, "ok": 0})
            bucket["total"] += 1
            if r["ok"]:
                bucket["ok"] += 1

    report = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "dataset": str(args.dataset),
        "dry_run": args.dry_run,
        "check_intent": args.check_intent,
        "total_casos": total,
        "casos_ok": passed,
        "tasa_exito": round(passed / total, 4) if total else 0.0,
        "por_tag": by_tag,
        "casos": [{"id": r["id"], "ok": r["ok"], "intencion_esperada": r["intencion_esperada"],
                   "intencion_detectada": r["intencion_detectada"], "checks": r["checks"],
                   **({"naturalidad": r["naturalidad"]} if "naturalidad" in r else {})}
                  for r in results],
    }

    naturalidades = [r["naturalidad"]["promedio"] for r in results if "naturalidad" in r]
    if naturalidades:
        report["naturalidad_promedio"] = round(sum(naturalidades) / len(naturalidades), 2)

    print(f"\nResultado: {passed}/{total} {f'({report["tasa_exito"] * 100:.1f}%)' if total else ''}")
    for tag in sorted(by_tag):
        b = by_tag[tag]
        print(f"  [{tag}] {b['ok']}/{b['total']}")
    if naturalidades and "naturalidad_promedio" in report:
        print(f"Naturalidad promedio (1-5): {report['naturalidad_promedio']}")
    failed = [r for r in results if not r["ok"]]
    if failed:
        print("\nNo pasan:")
        for r in failed:
            fails = [c["criterio"] for c in r["checks"] if not c["ok"]]
            print(f"  * {r['id']}: {fails}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\nReporte: {args.output}")
    return report


async def main(args: argparse.Namespace) -> int:
    if args.no_report:
        args.output = None

    judge = None
    if args.judge:
        if not _DEFAULT_JUDGE_KEY:
            print("--judge requiere GROQ_API_KEY en el .env. Se omite la dimensión de naturalidad.")
        else:
            judge = AsyncGroq(api_key=_DEFAULT_JUDGE_KEY)

    cases = load_cases(args.dataset)
    if args.tag:
        cases = [c for c in cases if args.tag in c.get("tags", [])]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("Sin casos después de aplicar filtros.")
        return 1

    print(f"Casos: {len(cases)} | endpoint: {args.endpoint} | dry_run: {args.dry_run}"
          f"{' | judge: ' + (args.judge_model or 'default') if judge else ''}")
    results: list[dict] = []
    for i, case in enumerate(cases, 1):
        result = await run_case(case, args.endpoint, args.timeout,
                                args.check_intent, args.dry_run,
                                judge=judge, judge_model=args.judge_model or _DEFAULT_JUDGE_MODEL)
        results.append(result)
        print(f"  [{i}/{len(cases)}] {'OK ' if result['ok'] else 'FAIL'} {result['id']}")
    summarize(results, args)
    return 0 if all(r["ok"] for r in results) else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval del golden set de eia-rag (dev-only, contra instancia local).",
    )
    parser.add_argument("--endpoint", default="http://localhost:8000/chat",
                        help="URL de POST /chat (por defecto: local).")
    parser.add_argument("--dataset", type=Path, default=GOLDEN_SET,
                        help="Ruta al golden set JSONL.")
    parser.add_argument("--output", type=Path, default=REPORT,
                        help="Ruta del reporte JSON.")
    parser.add_argument("--tag", help="Filtrar casos por un tag.")
    parser.add_argument("--limit", type=int, help="Correr solo los primeros N casos.")
    parser.add_argument("--timeout", type=float, default=80.0,
                        help="Timeout por request en segundos (Groq razona).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula respuestas con los criterios 'debe_incluir' (sin red).")
    parser.add_argument("--check-intent", action="store_true",
                        help="Añade chequeo duro de intención detectada == esperada.")
    parser.add_argument("--judge", action="store_true",
                        help="Añade puntuación LLM-as-judge de naturalidad/tono (1-5) por turno.")
    parser.add_argument("--judge-model", default=_DEFAULT_JUDGE_MODEL,
                        help="Modelo del juez de naturalidad (por defecto: GROQ_CHAT_MODEL).")
    parser.add_argument("--no-report", action="store_true", help="No escribir report.json.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args(sys.argv[1:]))))