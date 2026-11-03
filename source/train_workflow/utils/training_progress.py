"""Console/file progress for chunked PPO learning; no simulator dependencies.

Only logger presentation arguments are changed. Optimization, sampling and
iteration numbering stay under the existing runner/curriculum.
"""
from __future__ import annotations

import json
import copy
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path


STAGES = {"short": (1, "短片段学习"), "long": (2, "长片段衔接"), "full": (3, "整段训练")}


def stage_plan(total, initial_progress=0.):
    if total < 1 or not 0 <= initial_progress <= 1:
        raise ValueError("Invalid training budget or curriculum progress")
    cuts = [0]
    for boundary in (.55, .85):
        at = (math.ceil((boundary-initial_progress)/(1-initial_progress)*max(1,total-1))
              if initial_progress < boundary else 0)
        cuts.append(min(total,max(0,at)))
    cuts.append(total)
    return [{"stage": name, "begin": lo, "end": hi}
            for name,lo,hi in zip(STAGES,cuts,cuts[1:]) if hi > lo]


def duration_text(seconds):
    if seconds is None:
        return "估算中"
    value=max(0,int(seconds));hours,rest=divmod(value,3600);minutes,secs=divmod(rest,60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TrainingProgress:
    def __init__(self, logger, *, start, total, initial_progress, motion_seconds,
                 log_dir=None, update_interval=200, probe_interval=400, emit=print, clock=time.monotonic):
        self.logger,self.start,self.total=logger,int(start),int(total)
        self.initial_progress=float(initial_progress)
        self.plan=stage_plan(self.total,self.initial_progress)
        self.motion_seconds=float(motion_seconds)
        self.path=Path(log_dir)/"training_status.json" if log_dir is not None else None
        self.emit,self.clock=emit,clock
        self.update_interval,self.probe_interval=update_interval,probe_interval
        self.original_log=logger.log
        self.started=clock();self.done=0;self.learning_seconds=0.
        self.phase="initializing";self.stage=self.plan[0]["stage"]
        self.block_begin=self.block_end=0
        self.strength=self.learning_rate=None
        self.lr_policy=None
        self.next_probe=None;self.latest_checkpoint=None
        self.last_coverage=None;self.full_checks=[];self.validation=None
        self.validation_passed=None;self.error=None;self.io_warned=False
        self.quality_selection=None
        self.stop_reason=None
        self.overhead_seconds={};self._overhead_open=None
        self.expects_validation_video=False;self.expects_final_export=False

    def __enter__(self):
        self.logger.log=self.log
        self.emit(f"[训练计划] 本次预算上限 {self.total} 次 | 累计预算上限 {self.start+self.total} 次 "
                  f"| 迭代编号 {self.start}–{self.start+self.total-1}")
        self.emit(f"[训练计划] 每 {self.update_interval} 次更新采样 | "
                  f"覆盖验证间隔 {self.probe_interval if self.probe_interval else '关闭'} | "
                  f"进度按单次PPO迭代更新，{self.update_interval}次只是内部更新段")
        for item in self.plan:
            index,name=STAGES[item["stage"]]
            self.emit(f"[课程计划] {index}/3 {name} | 迭代编号 "
                      f"{self.start+item['begin']}–{self.start+item['end']-1} | {self.episode_text(item['stage'])}")
        self._persist()
        return self

    def __exit__(self, typ, error, traceback):
        self.end_overhead()
        self.logger.log=self.original_log
        if error is not None:
            if isinstance(error,KeyboardInterrupt):
                self.phase="stopped";self.stop_reason="manual_stop"
                best=(self.quality_selection or {}).get("best")
                detail=(f"已保存完成 {best['iteration']} 次的best部署包，可用于人工验收" if best else "尚无已确认的合格best")
                self.emit(f"[手动停止] 已完成 {self.done}/{self.total} 次；{detail}。当前未验收权重不会覆盖best。")
                self._persist()
                return False
            self.phase="failed";self.error=f"{typ.__name__}: {error}"
            self.emit(f"[运行异常] 本次学习完成 {self.done}/{self.total} | {self.error}")
            self._persist()
        return False

    def episode_text(self, stage=None):
        stage=stage or self.stage
        if stage=="full":
            return f"75%从头整段({self.motion_seconds:.2f}s) + 25%局部({min(12.,self.motion_seconds):.2f}s)"
        return f"回合 {min(3. if stage=='short' else 12.,self.motion_seconds):.2f}s"

    def snapshot(self):
        item=next(x for x in self.plan if x["stage"]==self.stage)
        stage_done=min(item["end"]-item["begin"],max(0,self.done-item["begin"]))
        eta=(self.learning_seconds/self.done*(self.total-self.done)) if self.done else None
        cycles=self.remaining_validation_cycles()
        overhead,pending=self.overhead_estimate(cycles)
        return dict(phase=self.phase,stage=self.stage,stage_name=STAGES[self.stage][1],
            iteration_index=self.start+self.done-1 if self.done else None,
            cumulative_completed=self.start+self.done,cumulative_target=self.start+self.total,
            invocation_completed=self.done,invocation_total=self.total,
            invocation_percent=100*self.done/self.total,
            stage_completed=stage_done,stage_total=item["end"]-item["begin"],
            block_completed=max(0,self.done-self.block_begin),block_total=self.block_end-self.block_begin,
            jump_strength=self.strength,learning_rate=self.learning_rate,learning_rate_policy=self.lr_policy,
            wall_elapsed_s=max(0.,self.clock()-self.started),learning_elapsed_s=self.learning_seconds,
            learning_eta_s=eta,overhead_eta_s=overhead,overhead_pending=pending,
            remaining_validation_cycles=cycles,
            total_eta_s=None if eta is None else eta+(overhead or 0.),
            eta_scope="remaining PPO learning plus measured validation, video and export still ahead",
            next_coverage_at_invocation_completed=self.next_probe,
            latest_checkpoint=self.latest_checkpoint,last_coverage=self.last_coverage,
            full_checks=self.full_checks,validation=self.validation,
            validation_passed=self.validation_passed,error=self.error,
            quality_selection=self.quality_snapshot(),stop_reason=self.stop_reason,
            stage_plan=self.plan,updated_at_utc=datetime.now(timezone.utc).isoformat())

    def _persist(self):
        if self.path is None:return
        temporary=self.path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(self.snapshot(),ensure_ascii=False,indent=2),encoding="utf-8")
            os.replace(temporary,self.path)
        except OSError as error:
            if not self.io_warned:
                self.emit(f"[进度文件] 写入失败，终端进度仍继续：{error}")
                self.io_warned=True

    def set_block(self, *, done, count, stage, strength, learning_rate, next_probe=None, lr_policy=None):
        self.phase="learning";self.stage=stage
        self.block_begin=done;self.block_end=done+count
        self.strength=float(strength);self.learning_rate=float(learning_rate);self.next_probe=next_probe
        self.lr_policy=dict(lr_policy) if lr_policy is not None else None
        index,name=STAGES[stage]
        self.emit(f"[开始学习] 阶段 {index}/3 {name} | 本段迭代编号 "
                  f"{self.start+done}–{self.start+done+count-1} | {self.episode_text()}")
        if self.lr_policy:
            policy=self.lr_policy
            self.emit(f"[学习率策略] {policy['schedule']} | 范围 [{policy['lower']:.2e}, {policy['upper']:.2e}] "
                      f"| KL目标={policy['desired_kl']:.4f}")
        self._persist()

    def log(self, **kwargs):
        shown=dict(kwargs)
        # RSL's footer divides cumulative learning time by iterations since
        # start_it. Both endpoints must span this invocation, not each chunk.
        shown.update(start_it=self.start,total_it=self.start+self.total)
        self.original_log(**shown)
        self.done=min(self.total,max(0,int(kwargs["it"])-self.start+1))
        self.learning_seconds+=float(kwargs["collect_time"])+float(kwargs["learn_time"])
        self.learning_rate=float(kwargs["learning_rate"])
        state=self.snapshot();index,name=STAGES[self.stage]
        self.emit(f"[全局进度] 累计 {state['cumulative_completed']}/{state['cumulative_target']} "
                  f"({100*state['cumulative_completed']/state['cumulative_target']:.1f}%) | "
                  f"本次 {self.done}/{self.total} ({state['invocation_percent']:.1f}%) | iter编号 {kwargs['it']}")
        self.emit(f"[当前阶段] {index}/3 {name} | 本次阶段 {state['stage_completed']}/{state['stage_total']} "
                  f"| 内部更新段 {state['block_completed']}/{state['block_total']} | {self.episode_text()}")
        self.emit(f"[训练状态] 跳跃目标强度 {100*self.strength:.0f}% | LR={self.learning_rate:.2e} | "
                  f"下次覆盖验证：{'关闭/无剩余' if self.next_probe is None else f'本次完成{self.next_probe}次时'}")
        self.emit_eta(state)
        self.emit_quality()
        self._persist()

    def status(self, phase, message):
        self.phase=phase;self.emit(f"[运行状态] {message}");self._persist()

    def checkpoint_saved(self, path):
        self.latest_checkpoint=str(Path(path).resolve())
        self.emit(f"[检查点] 已保存 {self.latest_checkpoint}")
        self._persist()

    def begin_overhead(self, kind):
        self.end_overhead()
        self._overhead_open=(kind,self.clock())

    def end_overhead(self, kind=None):
        if self._overhead_open is None:return
        open_kind,started=self._overhead_open
        if kind is not None and kind!=open_kind:return
        self._overhead_open=None
        self.record_overhead(open_kind,self.clock()-started)

    def record_overhead(self, kind, seconds):
        self.overhead_seconds.setdefault(kind,[]).append(max(0.,float(seconds)))

    def remaining_validation_cycles(self):
        # Probes of this invocation land on interval, 2*interval, ... and the
        # final check is separate unless one of those probes is exactly `total`.
        if self.phase!="learning":return 0
        interval=int(self.probe_interval or 0)
        periodic=0
        if interval>0 and self.next_probe is not None:
            cursor=int(self.next_probe)
            while cursor<=self.total and periodic<100000:
                periodic+=1
                cursor+=interval
        separate=interval<=0 or self.total%interval!=0
        return periodic+(1 if separate else 0)

    def overhead_estimate(self, cycles):
        pending=[];total=0.;measured=False
        def add(kind, count):
            nonlocal total,measured
            if count<=0:return
            samples=self.overhead_seconds.get(kind) or []
            if not samples:
                pending.append(kind);return
            recent=samples[-3:]
            total+=sum(recent)/len(recent)*count
            measured=True
        add("coverage",cycles)
        add("full",cycles)
        if self.expects_validation_video:add("video",cycles)
        if self.expects_final_export and self.phase=="learning":add("export",1)
        # Export is seconds; a missing sample must not hide a measured validation.
        if not measured or any(kind!="export" for kind in pending):return None,pending
        return total,pending

    def emit_eta(self, state):
        elapsed=duration_text(state["wall_elapsed_s"])
        learning,overhead,pending=state["learning_eta_s"],state["overhead_eta_s"],state["overhead_pending"]
        if learning is None:
            self.emit(f"[时间估算] 本次运行 {elapsed} | 剩余时间估算中")
            return
        if overhead is None and pending:
            self.emit(f"[时间估算] 本次运行 {elapsed} | "
                      f"学习剩余约 {duration_text(learning)}（验证/导出待首次验收后计入）")
            return
        if overhead is None:
            self.emit(f"[时间估算] 本次运行 {elapsed} | 学习剩余约 {duration_text(learning)}")
            return
        labels={"coverage":"覆盖验证","full":"整段验证","video":"验收录像","export":"导出"}
        included="验证/导出" if self.expects_final_export and "export" not in pending else "验证"
        suffix=""
        if pending:
            suffix="，"+"、".join(labels[kind] for kind in pending)+"耗时尚未计入"
        self.emit(f"[时间估算] 本次运行 {elapsed} | 剩余约 {duration_text(learning+overhead)}"
                  f"（学习 {duration_text(learning)} + {included} {duration_text(overhead)}{suffix}）")

    def validation_started(self, kind, seed, starts, seconds):
        self.phase="full_validation" if kind=="full" else "coverage_validation"
        self.next_probe=None
        self.begin_overhead("full" if kind=="full" else "coverage")
        self.validation=dict(kind=kind,seed=seed,starts=starts,episode_seconds=seconds)
        self.emit(f"[验证开始] {'整段' if kind=='full' else '中途重置诊断'} | seed={seed} | "
                  f"起点数={starts} | 回合={seconds:.2f}s | 学习进度 {self.done}/{self.total}")
        self._persist()

    def validation_finished(self, record):
        self.end_overhead()
        summary={key:record[key] for key in ("kind","seed","completed","tested")}
        summary.update(quality_completed=record.get("quality_completed"),quality_passed=record.get("quality_passed"))
        summary.update(acceptance_completed=record.get("acceptance_completed"),acceptance_passed=record.get("acceptance_passed"))
        if record["kind"]=="full":self.full_checks.append(summary)
        else:self.last_coverage=summary
        self.validation=dict(summary,state="finished")
        self.emit(f"[验证结束] {record['kind']} seed={record['seed']} | "
                  f"跑完 {record['completed']}/{record['tested']} | "
                  f"持续异常筛选 {record.get('quality_completed','未知')}/{record['tested']} | "
                  f"动作验收 {record.get('acceptance_completed','未知')}/{record['tested']}")
        self._persist()

    def quality_snapshot(self):
        if self.quality_selection is None:return None
        state=copy.deepcopy(self.quality_selection)
        iteration=(state.get("latest") or {}).get("iteration")
        state["iterations_since_validation"]=max(0,self.start+self.done-iteration) if iteration is not None else None
        return state

    def emit_quality(self):
        state=self.quality_snapshot()
        if state is None:return
        latest=state.get("latest");best=state.get("best")
        if latest is None:
            self.emit("[质量状态] 尚无本次已记录的定期验收结果，等待首次验收")
        else:
            confirmation="通过" if latest["confirmed"] is True else "未通过" if latest["confirmed"] is False else "尚未确认"
            score="未知" if latest.get("score") is None else f"{latest['score']:.4f}（越低越好）"
            self.emit(f"[质量状态] 最近验收：完成 {latest['iteration']} 次（距当前 {state['iterations_since_validation']} 次） | "
                      f"初验整段及窗口{'通过' if latest['screen_passed'] else '未通过'} | 三种子{confirmation} | "
                      f"连续达标 {state['streak']}/{state['required_streak']} | 最近评分 {score}")
            metrics=latest.get("metrics",{})
            if metrics:
                value=lambda key,unit:f"{metrics[key]:.3f}{unit}" if key in metrics else "未知"
                self.emit(f"[质量指标·最近验收] 腿P95={value('leg_p95','rad')} | 高度P95={value('height_p95','m')} | "
                          f"倾斜P95={value('tilt_p95','rad')} | 连续窗口 {latest.get('windows_completed')}/{latest.get('windows_tested')} | "
                          f"腾空事件 {latest.get('jump_events_passed',0)}/{latest.get('jump_events_tested',0)}（已验种子累计）")
            if latest.get("failures"):self.emit(f"[最近未通过原因] {latest['failures'][0]}")
            if "motion_score" in latest:
                motion="未知" if latest["motion_score"] is None else f"{latest['motion_score']:.3f}"
                windows=[round(w["start_s"],2) for w in latest.get("motion_review_windows",[])]
                self.emit(f"[运动诊断·最近验收] 参考对照指标={motion}（越低越好） | "
                          f"跨种子候选窗口={latest.get('motion_candidates',0)} | 优先人工查看起点秒数={windows}；候选不等于确定滑步")
        if best:
            recommendation=("最近已验收检查点达到稳定标准，可手动停止后人工验收best" if state["manual_review_ready"] else
                            "已有合格best可人工预览，最近检查尚未达到连续稳定标准")
            best_score=f"{best['score']:.4f}" if best.get("score") is not None else "未知"
            self.emit(f"[人工验收] {recommendation} | best：完成 {best['iteration']} 次，评分 {best_score} | "
                      "训练继续至设定上限或手动停止")
        else:self.emit("[人工验收] 尚无已确认的合格best，继续训练与定期验收")

    def quality_checked(self, selection, early_stop_enabled=False):
        self.quality_selection=dict(streak=selection.streak,required_streak=selection.patience,
            last_checked_completed_iteration=selection.last_iteration,best=selection.best,
            latest=getattr(selection,"latest",None),manual_review_ready=getattr(selection,"manual_review_ready",False),
            manual_review_available=selection.best is not None,
            stop_reason=selection.stop_reason,early_stop_enabled=False,profile_kind=selection.profile["kind"])
        self.quality_selection=copy.deepcopy(self.quality_selection)
        self.emit_quality()
        self._persist()

    def validation_progress(self, *, batch, batches, elapsed, length, pending):
        self.validation.update(batch=batch,batches=batches,episode_elapsed_s=elapsed,pending=pending)
        self.emit(f"[验证进行中] {self.validation['kind']} seed={self.validation['seed']} | "
                  f"批次 {batch}/{batches} | 回合时间 {elapsed:.2f}/{length:.2f}s | 待结束环境 {pending}")
        self._persist()

    def finish(self, passed):
        self.phase="completed";self.validation_passed=bool(passed)
        self.stop_reason=(self.quality_selection or {}).get("stop_reason")
        self.emit(f"[运行结束] 实际完成 {self.done} 次学习 / 本次预算上限 {self.total} 次，最终验证和导出已结束 | "
                  f"整段完成及质量验证{'通过' if passed else '未全部通过，请查看报告与未合格调试包'}")
        self._persist()
