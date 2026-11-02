"""subarashii CLI.

  subarashii dub /data/input/video01.mkv            # fully automatic
  subarashii dub video.mp4 --provider anthropic --set translation.honorifics=keep
  subarashii dub video.mp4 --review                      # stop before synthesis; continue in the web UI
  subarashii dub drama.mkv --lang ko                     # spoken language (default: detect)
  subarashii resume <job-id> [--from-stage translate]
  subarashii jobs
  subarashii serve
"""

import argparse
import shutil
import sys
from pathlib import Path

from .config import data_dir, deep_merge, load_config, overrides_from_pairs, set_dotted
from .lang import SUPPORTED
from .project import STAGES, Job


def _resolve_input(raw: str, cfg) -> Path:
    p = Path(raw)
    for cand in (p, data_dir(cfg, "input") / raw):
        if cand.is_file():
            return cand.resolve()
    sys.exit(f"input not found: {raw} (put videos in {data_dir(cfg, 'input')})")


def _find_job(cfg, jid: str) -> Job:
    jobs = data_dir(cfg, "jobs")
    matches = [d for d in jobs.iterdir() if d.name == jid] or [d for d in jobs.iterdir() if d.name.startswith(jid)]
    if len(matches) != 1:
        sys.exit(f"job not found or ambiguous: {jid}")
    return Job.load(matches[0])


def _job_overrides(args) -> dict:
    ov = overrides_from_pairs(args.set)
    if getattr(args, "lang", None) and args.lang != "auto":
        set_dotted(ov, "input.language", args.lang)
    if args.provider:
        set_dotted(ov, "llm.provider", args.provider)
    if args.model:
        provider = args.provider or load_config(ov).llm.provider
        set_dotted(ov, f"llm.{provider}.model", args.model)
    return ov


def cmd_dub(args):
    from . import pipeline

    ov = _job_overrides(args)
    cfg = load_config(ov)
    src = _resolve_input(args.input, cfg)
    job = Job.create(data_dir(cfg, "jobs"), src, review=args.review, overrides=ov)
    print(f"job {job.p.id}")
    status = pipeline.run(job)
    _report(job, status, args.output)


def cmd_resume(args):
    from . import pipeline

    cfg = load_config()
    job = _find_job(cfg, args.job)
    if args.set or args.provider or args.model:
        job.p.overrides = deep_merge(job.p.overrides, _job_overrides(args))
    if job.p.status == "review" or args.approve:
        pipeline.approve_review(job)
    status = pipeline.run(job, from_stage=args.from_stage)
    _report(job, status, args.output)


def _report(job: Job, status: str, output: str | None):
    if status == "done":
        if output:
            shutil.copy2(job.p.output, output)
            print(f"done: {output}")
        else:
            print(f"done: {job.p.output}")
    elif status == "review":
        print(f"paused for review. Open the web UI, or run: subarashii resume {job.p.id}")
    else:
        sys.exit(f"{status}: {job.p.error}")


def cmd_jobs(args):
    cfg = load_config()
    for d in sorted(data_dir(cfg, "jobs").iterdir()):
        if (d / "project.json").is_file():
            p = Job.load(d).p
            print(f"{p.id:60s} {p.status:9s} {p.stage:12s} {p.source_name}")


def cmd_serve(args):
    import uvicorn

    uvicorn.run("subarashii.web.server:app", host=args.host, port=args.port, log_level="info")


def main():
    ap = argparse.ArgumentParser(prog="subarashii", description="Japanese / Chinese / Korean -> English dubbing")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--provider", choices=["llamacpp", "openai", "anthropic"], help="LLM backend")
        p.add_argument("--model", help="LLM model name for the chosen provider")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
        p.add_argument("-o", "--output", help="copy the finished video here")

    p = sub.add_parser("dub", help="dub a video")
    p.add_argument("input", help="video path, or a file name inside /data/input")
    p.add_argument("--review", action="store_true", help="pause before synthesis for review in the web UI")
    p.add_argument("--lang", choices=["auto", *SUPPORTED], help="spoken language (default: detect)")
    common(p)
    p.set_defaults(func=cmd_dub)

    p = sub.add_parser("resume", help="continue or re-run a job")
    p.add_argument("job", help="job id (prefix ok)")
    p.add_argument("--from-stage", choices=STAGES)
    p.add_argument("--approve", action="store_true", help="mark review as done")
    common(p)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("jobs", help="list jobs")
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("serve", help="run the web UI")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
