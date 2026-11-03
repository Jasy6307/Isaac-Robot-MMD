"""Summarize pre/post rollout probes; optionally wait for a running resume job."""
import argparse
import json
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(run):
    before=read(run/"repair_before.json")
    after=read(run/"repair_after.json")
    full=read(run/"auto_validation.json")
    previous={(r["seed"],x["start_s"]):(r,x) for r in before for x in r["results"]}
    comparisons=[]
    for record in after:
        for item in record["results"]:
            old_record,old=previous[(record["seed"],item["start_s"])]
            def compact(r,x):
                expected=sum(b["expected_flight_samples"] for b in r["bins"]
                    if x["start_s"]-1 < b["start_s"] < x["start_s"]+3.)
                missed=sum(b["expected_flight_samples"]*(b["missed_flight_fraction"] or 0.) for b in r["bins"]
                    if x["start_s"]-1 < b["start_s"] < x["start_s"]+3.)
                return dict(completed=x["completed"],quality=x.get("quality"),
                    expected_flight_samples=expected,missed_flight_fraction=missed/expected if expected else None)
            a,b=compact(old_record,old),compact(record,item)
            old_pass=old["completed"] and old.get("quality",{}).get("passed") is True
            new_pass=item["completed"] and item.get("quality",{}).get("passed") is True
            comparisons.append(dict(seed=record["seed"],start_s=item["start_s"],before=a,after=b,
                regression=bool(old_pass and not new_pass),recovered=bool(not old_pass and new_pass)))
    return dict(status="completed",run=str(run.resolve()),
        full_validation_passed=all(r["completed"]==r["tested"] and r.get("quality_passed") is True for r in full),
        full_checks=[dict(seed=r["seed"],completed=r["completed"],tested=r["tested"],
            quality_completed=r.get("quality_completed"),results=r["results"]) for r in full],
        probe_comparisons=comparisons,
        note="Recorded completion/tracking criteria and flight checks; naturalness still needs visual review.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--watch",action="store_true")
    parser.add_argument("--console_log",type=Path)
    parser.add_argument("--timeout_hours",type=float,default=6.)
    args=parser.parse_args()
    deadline=time.monotonic()+args.timeout_hours*3600
    result=None
    while args.watch:
        path=args.run/"training_status.json"
        state=read(path) if path.exists() else {}
        if state.get("phase")=="completed":break
        if state.get("phase")=="failed":
            result=dict(status="failed",error=state.get("error"));break
        if args.console_log and args.console_log.exists():
            with args.console_log.open("rb") as stream:
                stream.seek(max(0,args.console_log.stat().st_size-12000))
                tail=stream.read().decode("utf-8",errors="replace")
            if "Traceback (most recent call last):" in tail:
                result=dict(status="failed",error="Training traceback; inspect the console log",console_tail=tail);break
        if time.monotonic()>deadline:
            result=dict(status="monitor_timeout",error="Training was not stopped; inspect its current status");break
        time.sleep(10.)
    if result is None:result=summarize(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 续训前后比较","",f"状态：{result['status']}",""]
    if result["status"]=="completed":
        lines += [f"三个种子的整段完成及质量验收：{result['full_validation_passed']}","",
            "| seed | 片段起点(s) | 续训前完成/质量 | 续训后完成/质量 | 已有合格项退步 |",
            "|---|---|---|---|---|"]
        for row in result["probe_comparisons"]:
            a,b=row["before"],row["after"]
            lines.append(f"| {row['seed']} | {row['start_s']:.2f} | {a['completed']}/{a['quality']['passed']} "
                         f"| {b['completed']}/{b['quality']['passed']} | {row['regression']} |")
        lines += ["","跳跃的预计腾空样本与未完成比例、各窗口误差峰值和持续时间详见同名JSON。",
            "上述自动指标不能替代视频复核；不会因为跑完就标为质量通过。"]
    else:lines.append(str(result.get("error")))
    args.output.with_suffix(".md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("probe_comparisons","full_checks")},ensure_ascii=False))


if __name__=="__main__":main()
