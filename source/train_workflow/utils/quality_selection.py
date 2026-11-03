"""Frozen reference-specific quality gates and checkpoint selection (CPU only).

Contact-free motion is mandatory only for strong reference events. Unknown
reference intent never becomes a hard support label. Thresholds are engineering
acceptance criteria, not a proof of choreography or aesthetic quality.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


TRACE_FIELDS = ("seconds", "leg", "height", "tilt", "normal", "required",
                "unexpected_air", "contact_left", "contact_right", "qualified",
                "vz", "feet_distance", "support_speed", "excess_height", "excess_vz")
INDEX = {name: i for i, name in enumerate(TRACE_FIELDS)}
DEFAULT_LIMITS = dict(leg_p95=.30, height_p95=.12, tilt_p95=.35,
    leg_window=.35, height_window=.15, tilt_window=.45,
    support_air_s=.10, support_air_fraction=.05, support_speed_p95=.80,support_speed_window=.80,
    excess_height_window=.10, excess_vz_window=.60,
    takeoff_offset_s=.20, landing_offset_s=.25)
PROFILE_DIR = Path(__file__).resolve().parents[1] / "quality_profiles"


def reference_key(report):
    contract = {k: report.get(k) for k in ("source_sha256", "duration_s", "control_hz",
                "version", "fps", "events", "support_feet", "support_windows")}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()


def generic_profile(report):
    return dict(version=1, reference_key=reference_key(report), kind="generic_engineering",
                limits=dict(DEFAULT_LIMITS), windows={}, events={}, provenance=[])


def resolve_profile(report, path=None):
    key = reference_key(report)
    target = Path(path) if path else PROFILE_DIR / f"{key}.json"
    profile = json.loads(target.read_text(encoding="utf-8")) if target.exists() else generic_profile(report)
    if path and not target.exists():
        raise FileNotFoundError(target)
    return validate_profile(report,profile)


def validate_profile(report,profile):
    if profile.get("version") != 1 or profile.get("reference_key") != reference_key(report):
        raise ValueError("Quality profile differs from the reference, event/support analysis, window or control rate")
    limits = profile.get("limits", {})
    if set(limits) != set(DEFAULT_LIMITS) or any(not math.isfinite(v) or v <= 0 for v in limits.values()):
        raise ValueError("Incomplete or invalid quality limits")
    for section in ("windows","events"):
        if any(not math.isfinite(v) or v<=0 for row in profile.get(section,{}).values() for v in row.values()):
            raise ValueError("Invalid local/event quality limits")
    return profile


def _longest(mask, seconds, dt):
    longest = run = 0
    for i, value in enumerate(mask):
        contiguous = i == 0 or seconds[i]-seconds[i-1] <= 1.5*dt
        run = (run+1 if contiguous else 1) if value else 0
        longest = max(longest, run)
    return longest*dt


def _p95(values):
    return float(np.quantile(values, .95)) if len(values) else None


def assess_trace(trace, report, profile, *, full=False):
    """Assess one first-episode trace. Missing/nonfinite observations fail closed."""
    data = np.asarray(trace, dtype=np.float64)
    dt = 1./report["control_hz"]
    failures = []
    result = dict(passed=False, failures=failures, metrics={}, windows=[], events=[])
    if data.ndim != 2 or data.shape[1] != len(TRACE_FIELDS) or not len(data) or not np.isfinite(data).all():
        failures.append(dict(check="missing_or_nonfinite_trace")); return result
    t = data[:, 0]
    if np.any(np.diff(t) < .5*dt) or np.any(np.diff(t) > 1.5*dt):
        failures.append(dict(check="missing_or_duplicate_control_steps")); return result
    if full and (t[0]>dt+1.e-5 or t[-1]+dt<report["duration_s"]-1.e-5):
        failures.append(dict(check="incomplete_full_trace",time_s=float(t[-1])))
    limits = profile["limits"]
    get = lambda name: data[:, INDEX[name]]
    normal = get("normal") >= .99
    support = get("required") > .5
    metrics = result["metrics"]
    metrics.update(leg_p95=_p95(get("leg")[normal]), height_p95=_p95(get("height")),
        tilt_p95=_p95(get("tilt")), support_air_s=_longest(support & (get("unexpected_air") > .5),t,dt),
        support_air_fraction=float(get("unexpected_air")[support].mean()) if support.any() else None,
        support_speed_p95=_p95(get("support_speed")[support]))
    def check(name, value, limit, seconds=None):
        if value is not None and value > limit+1.e-7:
            failures.append(dict(check=name, observed=float(value), limit=float(limit), time_s=seconds))
    for name, value in metrics.items(): check(name,value,limits[name])
    # Half-second grid, overlapping one-second windows. Partial boundary windows
    # remain visible, but use the generic limit unless the baseline covers them.
    for start in np.arange(math.floor(t[0]*2)/2, t[-1]+dt/2, .5):
        inside = (t >= start-1.e-5) & (t < start+1.-1.e-5)
        if inside.sum() < max(2,round(.2/dt)): continue
        row = dict(start_s=float(start), end_s=float(min(start+1.,t[-1]+dt)), samples=int(inside.sum()))
        for name, column, mask in (("leg_window","leg",normal),("height_window","height",np.ones(len(t),bool)),
                ("tilt_window","tilt",np.ones(len(t),bool)),("excess_height_window","excess_height",normal),
                ("excess_vz_window","excess_vz",normal),("support_speed_window","support_speed",support)):
            selected = inside & mask
            value = float(get(column)[selected].mean()) if selected.sum() >= max(2,round(.2/dt)) else None
            row[name] = value
            local = profile.get("windows", {}).get(f"{start:.1f}", {})
            # A short reset fragment must not be compared with an entire
            # baseline window; the global strict ceiling still applies.
            calibrated = local.get(name,limits[name]) if inside.sum() >= round(.9/dt) else limits[name]
            check(name,value,min(limits[name],calibrated),float(start))
        result["windows"].append(row)
    for event in report["events"]:
        if not event["flight_reward"]: continue
        phases=event["phases"];lo,hi=phases[0][0],phases[3][1]
        complete=t[0] <= lo+dt+1.e-5 and t[-1]+dt >= hi-1.e-5
        row=dict(id=event["id"],start_s=lo,end_s=hi,assessed=bool(complete))
        result["events"].append(row)
        if not complete:
            if full: failures.append(dict(check="unobserved_jump_event",time_s=lo,event_id=event["id"]))
            continue
        inside=(t>=lo-1.e-5)&(t<hi-1.e-5)
        core=(t>=phases[2][0]-1.e-5)&(t<phases[2][1]-1.e-5)
        air=(get("contact_left")<.5)&(get("contact_right")<.5)
        qualified=core & inside & (get("qualified")>.5)
        # Find an actual consecutive flight segment overlapping qualified
        # reference flight. Ground contact in transition frames is allowed.
        segments=[];begin=None
        for i,value in enumerate(air & inside):
            if value and begin is None: begin=i
            if begin is not None and (not value or i==len(t)-1):
                end=i if not value else i+1
                if (end-begin)*dt >= 2*dt-1.e-6 and qualified[begin:end].any(): segments.append((begin,end))
                begin=None
        row["flight_passed"]=bool(segments)
        if not segments:
            failures.append(dict(check="missed_sustained_flight",time_s=phases[2][0],event_id=event["id"]))
            continue
        begin,end=max(segments,key=lambda pair:pair[1]-pair[0])
        row.update(air_s=(end-begin)*dt,takeoff_offset_s=float(t[begin]-phases[2][0]),
                   landing_offset_s=float(t[end-1]+dt-phases[2][1]),
                   min_feet_distance_m=float(get("feet_distance")[begin:end].min()))
        landing=(t>=max(phases[3][0],t[end-1]))&(t<hi+1.e-5)
        supported=(get("contact_left")>.5)|(get("contact_right")>.5)
        row["landing_supported_s"]=_longest(landing & supported,t,dt)
        if row["landing_supported_s"] < 2*dt-1.e-6:
            failures.append(dict(check="landing_without_sustained_support",time_s=phases[3][0],event_id=event["id"]))
        event_limits=profile.get("events",{}).get(str(event["id"]),{})
        for name in ("takeoff_offset_s","landing_offset_s"):
            check(name,abs(row[name]),event_limits.get(name,limits[name]),phases[2][0])
        # Closing is constrained only for reference closing events with an
        # accepted, reference-specific baseline. Other jumps need not close.
        if event.get("evidence",{}).get("closing_mps",0.) > .2 and "min_feet_distance_m" in event_limits:
            check("jump_closing_distance",row["min_feet_distance_m"],event_limits["min_feet_distance_m"],phases[2][0])
    # Ranking emphasizes the worst local window; long easy regions cannot
    # compensate for a bad second. Only qualified candidates may be selected.
    local_scores=[sum(w[k]/DEFAULT_LIMITS[k] for k in ("leg_window","height_window","tilt_window") if w[k] is not None)/3
                  for w in result["windows"]]
    result["score"]=max(local_scores,default=float("inf")) + .25*float(np.mean(local_scores)) if local_scores else None
    result["passed"]=not failures
    return result


def calibrate_profile(report, records, provenance):
    """Freeze allowances using explicitly accepted rollouts, never a trainee."""
    profile=generic_profile(report)
    profile.update(kind="accepted_baseline",provenance=provenance)
    strict=[r["acceptance"] for record in records for r in record["results"]]
    if not strict or any(not r["completed"] or not r["quality"]["passed"] for record in records for r in record["results"]):
        raise ValueError("Calibration requires complete accepted rollouts without sustained tracking failures")
    margins=dict(leg_p95=.04,height_p95=.02,tilt_p95=.06,support_air_s=1./report["control_hz"],
                 support_air_fraction=.015,support_speed_p95=.08)
    for key,margin in margins.items():
        values=[s["metrics"][key] for s in strict if s["metrics"].get(key) is not None]
        if values: profile["limits"][key]=max(values)*1.15+margin
    window_margins=dict(leg_window=.04,height_window=.02,tilt_window=.06,excess_height_window=.02,excess_vz_window=.12,
                        support_speed_window=.08)
    for s in strict:
        for window in s["windows"]:
            if window["samples"] < round(.9*report["control_hz"]):continue
            row=profile["windows"].setdefault(f"{window['start_s']:.1f}",{})
            for key,margin in window_margins.items():
                if window[key] is not None:row[key]=max(row.get(key,0.),window[key]*1.15+margin)
        for event in s["events"]:
            if not event["assessed"] or not event.get("flight_passed") or event.get("landing_supported_s",0.) < 2./report["control_hz"]-1.e-6:
                raise ValueError(f"Accepted baseline has missing flight/landing evidence: {event}")
            row=profile["events"].setdefault(str(event["id"]),{})
            for key in ("takeoff_offset_s","landing_offset_s"):
                row[key]=max(row.get(key,0.),abs(event[key])+2./report["control_hz"])
            row["min_feet_distance_m"]=max(row.get("min_feet_distance_m",0.),event["min_feet_distance_m"]+.03)
    # Global strict ceilings must include the calibrated accepted envelope.
    for key in window_margins:
        profile["limits"][key]=max([DEFAULT_LIMITS[key],*[row.get(key,0.) for row in profile["windows"].values()]])
    return profile


def continuous_coverage(full_record):
    """Local acceptance must describe runtime continuity, not zero-velocity resets.

    The full trace gate checks the clock for gaps, reference duration and every
    overlapping local window. This receipt makes that coverage explicit.
    """
    count=sum(len(r.get("acceptance",{}).get("windows",[])) for r in full_record.get("results",[]))
    passed=(full_record.get("kind")=="full" and count>0 and full_record.get("acceptance_passed") is True
            and full_record.get("completed")==full_record.get("tested"))
    return dict(kind="coverage",scope="continuous_full_motion_windows",seed=full_record.get("seed"),
        completed=count if passed else 0,tested=count,quality_passed=full_record.get("quality_passed",False),
        acceptance_passed=passed,acceptance_score=full_record.get("acceptance_score"))


class QualitySelection:
    """Periodic confirmation and best-model selection, with resumable state."""
    def __init__(self, profile, *, patience=3, interval=400, state=None):
        self.profile=profile;self.patience=int(patience);self.interval=int(interval)
        if self.patience<1 or self.interval<1:raise ValueError("Quality patience/interval must be positive")
        self.streak=0;self.last_iteration=None;self.best=None;self.history=[];self.stop_reason=None
        self.latest=None
        if state:
            if state.get("version")!=1 or state.get("profile")!=profile or state.get("patience")!=patience or state.get("interval")!=interval:
                raise ValueError("Resumed quality selection settings differ from saved settings")
            for name in ("streak","last_iteration","best","history","stop_reason","latest"):
                setattr(self,name,copy.deepcopy(state.get(name,getattr(self,name))))
            self.stop_reason=None

    @staticmethod
    def passed(records):
        return bool(records) and all(r.get("quality_passed") is True and r.get("acceptance_passed") is True
            and r.get("completed")==r.get("tested") and r.get("tested",0)>0
            and isinstance(r.get("acceptance_score"),(int,float)) and math.isfinite(r["acceptance_score"]) for r in records)

    @staticmethod
    def score(records):
        return max(r["acceptance_score"] for r in records)

    @staticmethod
    def motion_score(records):
        if not records or not all(r.get("motion_feedback_required") for r in records):return None
        values=[]
        for record in records:
            if not record.get("results"):return None
            for trial in record["results"]:
                motion=trial.get("motion_diagnostics",{})
                value=motion.get("score")
                if motion.get("protocol")!="reference-motion-v1" or motion.get("complete") is not True or motion.get("contact_observed") is not True or motion.get("audit_only") is True or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:return None
                values.append(value)
        return max(values) if values else None

    @property
    def manual_review_ready(self):
        return bool(self.latest and self.latest["screen_passed"] and self.latest["confirmed"] is True
                    and self.streak>=self.patience)

    @staticmethod
    def details(records):
        """Compact display evidence; never interpret missing metrics as a pass."""
        metrics={};failures=[];events=[]
        for record in records:
            if record is None:continue
            for result in record.get("results",[]):
                acceptance=result.get("acceptance",{})
                for key in ("leg_p95","height_p95","tilt_p95"):
                    value=acceptance.get("metrics",{}).get(key)
                    if isinstance(value,(int,float)) and math.isfinite(value):metrics[key]=max(metrics.get(key,value),value)
                events.extend(e for e in acceptance.get("events",[]) if e.get("assessed"))
                failures.extend(dict(f,seed=record.get("seed")) for f in acceptance.get("failures",[]))
                quality=result.get("quality",{})
                if quality.get("passed") is False:
                    failures.append(dict(check="sustained_tracking_failure",time_s=quality.get("first_issue_s"),seed=record.get("seed")))
            if record.get("completed",0)<record.get("tested",0):
                failures.append(dict(check="incomplete_rollout",seed=record.get("seed")))
        details=dict(metrics=metrics,failures=failures[:3],jump_events_tested=len(events),
                    jump_events_passed=sum(bool(e.get("flight_passed")) for e in events))
        if any(r and r.get("motion_feedback_required") for r in records):
            details["motion_score"]=QualitySelection.motion_score(records)
            details["motion_candidates"]=sum(len(t.get("motion_diagnostics",{}).get("candidates",[])) for r in records if r for t in r.get("results",[]))
            candidates=[dict(c,seed=r["seed"]) for r in records if r for t in r.get("results",[])
                        for c in t.get("motion_diagnostics",{}).get("candidates",[])]
            details["motion_review_windows"]=sorted(candidates,key=lambda c:c["severity"],reverse=True)[:3]
            if details["motion_score"] is None:details["failures"].append(dict(check="missing_motion_diagnostics"))
        return details

    def screen(self, iteration, coverage, full):
        if self.last_iteration is not None and iteration<=self.last_iteration:
            raise ValueError("Quality checks must advance iteration counts")
        passed=(self.passed([coverage,full]) and coverage.get("kind")=="coverage" and full.get("kind")=="full"
                and coverage.get("seed")==full.get("seed")) if full is not None else False
        contiguous=self.last_iteration is None or iteration-self.last_iteration==self.interval
        self.streak=(self.streak+1 if contiguous else 1) if passed else 0
        self.last_iteration=iteration
        self.latest=dict(iteration=iteration,screen_passed=passed,confirmed=None,
            score=full.get("acceptance_score") if full else None,
            windows_completed=coverage.get("completed",0),windows_tested=coverage.get("tested",0),
            **self.details([full]))
        self.history.append(dict(iteration=iteration,passed=passed,streak=self.streak,
            score=self.score([full]) if passed else None))
        return passed and self.streak>=self.patience

    def confirm(self, iteration, records, seeds, checkpoint):
        if len(records)!=len(seeds) or [r.get("seed") for r in records]!=list(seeds) or any(r.get("kind")!="full" for r in records):
            raise ValueError("Confirmations must contain every requested full-rollout seed")
        passed=self.passed(records)
        motion_required=any(r.get("motion_feedback_required") for r in records)
        motion_score=self.motion_score(records) if motion_required else None
        if motion_required and motion_score is None:passed=False
        if self.latest is not None and self.latest["iteration"]==iteration:
            passed=passed and self.latest["screen_passed"]
        if self.latest is None or self.latest["iteration"]!=iteration:
            self.latest=dict(iteration=iteration,screen_passed=passed,confirmed=None,score=None,
                             windows_completed=None,windows_tested=None)
        self.latest.update(confirmed=passed,score=self.score(records) if passed else None,**self.details(records))
        updated=False
        if passed:
            score=self.score(records)
            # Scores within 2% count as equivalent; retain the earlier model.
            improve=self.best is None or score<self.best["score"]*.98
            if self.best is not None and motion_required and self.best.get("motion_score") is not None:
                old=self.best["motion_score"]
                # New local gains cannot trade away the original full-motion gates.
                # Small equivalent score fluctuations retain the existing 2% band.
                improve=(improve and motion_score<=old*1.10+.05) or (
                    score<=self.best["score"]*1.02 and motion_score<old*.90 and old-motion_score>.05)
            if improve:
                self.best=dict(iteration=iteration,checkpoint=str(checkpoint),score=score,seeds=list(seeds))
                if motion_required:self.best.update(motion_score=motion_score,motion_protocol="reference-motion-v1")
                updated=True
        else:self.streak=0
        self.history.append(dict(iteration=iteration,confirmation=True,passed=passed,updated_best=updated))
        return passed,updated

    def state_dict(self):
        return dict(version=1,profile=self.profile,patience=self.patience,interval=self.interval,
            streak=self.streak,last_iteration=self.last_iteration,best=self.best,history=self.history,
            latest=self.latest,stop_reason=self.stop_reason)
