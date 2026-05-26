use anyhow::{Context, Result};
use clap::Parser;
use skillopt::adapter::{build_adapter, Adapter};
use skillopt::config::load_config;
use skillopt::data::{load_split, sample_batch};
use skillopt::gate::{evaluate_gate, GateInput};
use skillopt::gradient::{
    apply_patch, full_rewrite, merge_patches, rank_and_select, record_rejected, reflect, ReflectBatch,
};
use skillopt::memory::{replace_slow_update_field, run_meta_skill, run_slow_update};
use skillopt::openai::OpenAIClient;
use skillopt::scheduler::{autonomous_select, compute_lr, LrSchedule};
use skillopt::scoring::{mean, skill_hash};
use skillopt::types::{Edit, GateDecision, RuntimeState, StepBuffer, StepRecord, Trajectory};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[derive(Parser)]
struct Cli {
    #[arg(long)] config: PathBuf,
    #[arg(long)] split_dir: PathBuf,
    #[arg(long, default_value = "outputs/run")] out_root: PathBuf,
    #[arg(long, default_value = "train")] train_split: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("skillopt=info,info").init();

    let cli = Cli::parse();
    let cfg = load_config(&cli.config)?;
    let client = Arc::new(OpenAIClient::from_env()?);
    let adapter: Box<dyn Adapter> = build_adapter(&cfg.env.name);

    std::fs::create_dir_all(cli.out_root.join("skills"))?;
    std::fs::create_dir_all(cli.out_root.join("steps"))?;

    let train_items = load_split(&cli.split_dir, &cli.train_split)?;
    let sel_items = load_split(&cli.split_dir, &cfg.evaluation.sel_split)?;

    let train_pool = sample_batch(&train_items, cfg.train.train_size, 42);

    // === Resume detection ===
    let resume = try_load_resume(&cli.out_root)?;
    let mut skill;
    let mut best_skill;
    let mut current_score;
    let mut best_score;
    let mut history: Vec<StepRecord>;
    let mut step_idx: u32;
    let starting_epoch: u32;

    if let Some((state, hist)) = resume {
        skill = std::fs::read_to_string(cli.out_root.join(&state.current_skill_path))
            .with_context(|| format!("read resume current_skill {}", state.current_skill_path))?;
        best_skill = std::fs::read_to_string(cli.out_root.join(&state.best_skill_path))
            .unwrap_or_else(|_| skill.clone());
        current_score = state.current_score;
        best_score = state.best_score;
        history = hist;
        step_idx = state.step;
        starting_epoch = state.epoch.max(1);
        info!("RESUME at step={} epoch={} current={:.3} best={:.3}",
            step_idx, starting_epoch, current_score, best_score);
    } else {
        skill = std::fs::read_to_string(&cfg.env.skill_init)
            .with_context(|| format!("read skill_init {}", cfg.env.skill_init))?;
        best_skill = skill.clone();
        let _ = std::fs::write(cli.out_root.join("skills/skill_v0000.md"), &skill);

        let baseline = adapter
            .rollout(&client, &skill, &sel_items, cfg.evaluation.workers, cfg.model.temperature_target)
            .await?;
        current_score = mean(&baseline.iter().map(|t| t.score).collect::<Vec<_>>());
        best_score = current_score;
        history = vec![];
        step_idx = 0;
        starting_epoch = 1;
        info!("baseline sel_score = {:.3}", current_score);
    }

    let sched = LrSchedule::from_str(&cfg.optimizer.lr_schedule);
    let steps_per_epoch = ((cfg.train.train_size as f32) / (cfg.train.batch_size * cfg.train.accumulation) as f32).ceil() as u32;
    let total_steps = cfg.train.num_epochs * steps_per_epoch.max(1);

    // Load latest meta-skill memo (cross-run persistence)
    let mut meta_memo = load_latest_meta(&cli.out_root).unwrap_or_default();
    if !meta_memo.is_empty() {
        info!("loaded prior meta-skill memo ({} chars)", meta_memo.len());
    }

    let mut prev_epoch_skill = skill.clone();

    for epoch in starting_epoch..=cfg.train.num_epochs {
        let mut step_buffer = StepBuffer::default();
        info!("=== epoch {epoch}/{} ===", cfg.train.num_epochs);

        for s in 0..steps_per_epoch {
            step_idx += 1;
            let seed = (epoch as u64) * 1000 + s as u64;
            let train_batch = sample_batch(&train_pool, cfg.train.batch_size, seed);

            // 1. Rollout
            let rolls = adapter
                .rollout(&client, &skill, &train_batch, cfg.evaluation.workers, cfg.model.temperature_target)
                .await?;
            let train_score = mean(&rolls.iter().map(|t| t.score).collect::<Vec<_>>());

            let (succ, fail): (Vec<Trajectory>, Vec<Trajectory>) =
                rolls.into_iter().partition(|t| t.score > 0.5);

            // 2. Reflect on success and failure minibatches separately
            let mut all_failure_edits: Vec<Edit> = vec![];
            let mut all_success_edits: Vec<Edit> = vec![];

            for chunk in fail.chunks(cfg.gradient.minibatch_size) {
                let (edits, patterns) = reflect(
                    &client, &skill,
                    &ReflectBatch { trajectories: chunk, kind: "failure" },
                    &step_buffer,
                    &meta_memo,
                    cfg.model.temperature_optimizer,
                    &cfg.model.reasoning_effort,
                ).await?;
                for p in patterns {
                    if !step_buffer.failure_patterns.iter().any(|x| x == &p) {
                        step_buffer.failure_patterns.push(p);
                    }
                }
                all_failure_edits.extend(edits);
            }
            for chunk in succ.chunks(cfg.gradient.minibatch_size) {
                let (edits, _) = reflect(
                    &client, &skill,
                    &ReflectBatch { trajectories: chunk, kind: "success" },
                    &step_buffer,
                    &meta_memo,
                    cfg.model.temperature_optimizer,
                    &cfg.model.reasoning_effort,
                ).await?;
                all_success_edits.extend(edits);
            }

            // 3. Aggregate
            let merged = merge_patches(all_failure_edits, all_success_edits);
            let n_proposed = merged.len() as u32;

            // 4. Select under edit budget (cosine/linear/constant) OR autonomous
            let lr = compute_lr(&sched, step_idx - 1, total_steps, cfg.optimizer.learning_rate);
            let patch = if matches!(sched, LrSchedule::Autonomous) {
                autonomous_select(merged, 1.5, cfg.optimizer.learning_rate.max(8))
            } else {
                rank_and_select(merged, lr)
            };
            let n_selected = patch.len() as u32;

            // 5. Update — bounded patch OR full rewrite
            let do_full = cfg.optimizer.full_rewrite_every > 0
                && step_idx % cfg.optimizer.full_rewrite_every == 0;
            let candidate = if do_full {
                info!("step {step_idx}: full-rewrite path");
                full_rewrite(&client, &skill, &fail, &succ, &step_buffer,
                    cfg.model.temperature_optimizer, &cfg.model.reasoning_effort).await?
            } else {
                apply_patch(&skill, &patch)
            };
            let cand_hash = skill_hash(&candidate);

            // 6. Gate
            let cand_traj = adapter
                .rollout(&client, &candidate, &sel_items, cfg.evaluation.workers, cfg.model.temperature_target)
                .await?;
            let cand_score = mean(&cand_traj.iter().map(|t| t.score).collect::<Vec<_>>());

            let gate = evaluate_gate(&GateInput {
                candidate_score: cand_score,
                current_score,
                best_score,
                patch: &patch,
            });

            match gate.decision {
                GateDecision::AcceptNewBest => {
                    skill = candidate;
                    current_score = cand_score;
                    best_skill = skill.clone();
                    best_score = cand_score;
                    info!("step {step_idx}: ACCEPT new best lr={lr} sel={cand_score:.3}");
                }
                GateDecision::Accept => {
                    skill = candidate;
                    current_score = cand_score;
                    if cand_score >= best_score {
                        best_skill = skill.clone();
                        best_score = cand_score;
                    }
                    info!("step {step_idx}: ACCEPT lr={lr} sel={cand_score:.3}");
                }
                GateDecision::Reject => {
                    if cfg.optimizer.use_rejected_buffer {
                        record_rejected(&mut step_buffer, patch.clone(), gate.score_drop,
                            format!("step {step_idx} drop {:.3}", gate.score_drop));
                    }
                    info!("step {step_idx}: REJECT lr={lr} sel={cand_score:.3} (drop {:.3})", gate.score_drop);
                }
            }

            let rec = StepRecord {
                step: step_idx,
                epoch,
                lr_budget: lr,
                train_score,
                sel_score: cand_score,
                best_sel_score: best_score,
                gate: match gate.decision {
                    GateDecision::AcceptNewBest => GateDecision::AcceptNewBest,
                    GateDecision::Accept => GateDecision::Accept,
                    GateDecision::Reject => GateDecision::Reject,
                },
                n_proposed,
                n_selected,
                skill_hash: cand_hash,
            };
            history.push(rec.clone());
            persist_step(&cli.out_root, step_idx, epoch, current_score, best_score, &skill, &best_skill, &rec, &history)?;
        }

        // End-of-epoch hooks
        if cfg.optimizer.use_slow_update && epoch >= 2 {
            let pairs = build_pairs(&adapter, &client, &prev_epoch_skill, &skill, &train_pool, &cfg).await?;
            let block = run_slow_update(&client, &pairs, cfg.model.temperature_optimizer, &cfg.model.reasoning_effort).await?;
            if !block.trim().is_empty() {
                skill = replace_slow_update_field(&skill, &block);
                best_skill = replace_slow_update_field(&best_skill, &block);
                info!("epoch {epoch}: slow update injected ({} chars)", block.len());
            }
        }
        if cfg.optimizer.use_meta_skill && epoch >= 2 {
            let pairs = build_pairs(&adapter, &client, &prev_epoch_skill, &skill, &train_pool, &cfg).await?;
            let memo = run_meta_skill(&client, &pairs, cfg.model.temperature_optimizer, &cfg.model.reasoning_effort).await?;
            std::fs::create_dir_all(cli.out_root.join(format!("meta_skill/epoch_{epoch:02}")))?;
            std::fs::write(cli.out_root.join(format!("meta_skill/epoch_{epoch:02}/memo.md")), &memo)?;
            meta_memo = memo;
        }
        prev_epoch_skill = skill.clone();
    }

    std::fs::write(cli.out_root.join("best_skill.md"), &best_skill)?;
    println!("\nbest_sel_score = {:.3}", best_score);
    println!("artifact: {}", cli.out_root.join("best_skill.md").display());
    Ok(())
}

fn load_latest_meta(out_root: &std::path::Path) -> Option<String> {
    let dir = out_root.join("meta_skill");
    if !dir.exists() { return None; }
    let mut best: Option<(u32, std::path::PathBuf)> = None;
    for ent in std::fs::read_dir(&dir).ok()? {
        let ent = ent.ok()?;
        let name = ent.file_name();
        let s = name.to_string_lossy();
        if let Some(rest) = s.strip_prefix("epoch_") {
            if let Ok(n) = rest.parse::<u32>() {
                let memo = ent.path().join("memo.md");
                if memo.exists() && best.as_ref().map_or(true, |(b, _)| n > *b) {
                    best = Some((n, memo));
                }
            }
        }
    }
    best.and_then(|(_, p)| std::fs::read_to_string(p).ok())
}

fn try_load_resume(out_root: &std::path::Path) -> Result<Option<(RuntimeState, Vec<StepRecord>)>> {
    let state_path = out_root.join("runtime_state.json");
    if !state_path.exists() { return Ok(None); }
    let state: RuntimeState = serde_json::from_str(&std::fs::read_to_string(&state_path)?)?;
    if state.step == 0 { return Ok(None); }
    let hist_path = out_root.join("history.jsonl");
    let mut hist = vec![];
    if hist_path.exists() {
        for line in std::fs::read_to_string(&hist_path)?.lines() {
            if line.trim().is_empty() { continue; }
            if let Ok(rec) = serde_json::from_str::<StepRecord>(line) {
                hist.push(rec);
            }
        }
    }
    Ok(Some((state, hist)))
}

async fn build_pairs(
    adapter: &Box<dyn Adapter>,
    client: &OpenAIClient,
    prev: &str,
    curr: &str,
    pool: &[skillopt::types::TaskItem],
    cfg: &skillopt::config::Config,
) -> Result<Vec<(Trajectory, Trajectory)>> {
    let sample = sample_batch(pool, cfg.gradient.merge_batch_size.min(pool.len()), 7);
    let prev_t = adapter.rollout(client, prev, &sample, cfg.evaluation.workers, cfg.model.temperature_target).await?;
    let curr_t = adapter.rollout(client, curr, &sample, cfg.evaluation.workers, cfg.model.temperature_target).await?;
    let mut pairs = vec![];
    for (a, b) in prev_t.into_iter().zip(curr_t.into_iter()) {
        if a.item_id == b.item_id { pairs.push((a, b)); }
    }
    Ok(pairs)
}

fn persist_step(
    out_root: &std::path::Path,
    step: u32,
    epoch: u32,
    current_score: f32,
    best_score: f32,
    skill: &str,
    best_skill: &str,
    rec: &StepRecord,
    history: &[StepRecord],
) -> Result<()> {
    let dir = out_root.join(format!("steps/step_{:04}", step));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("record.json"), serde_json::to_string_pretty(rec)?)?;
    let cur_path = format!("skills/skill_v{:04}.md", step);
    std::fs::write(out_root.join(&cur_path), skill)?;
    std::fs::write(out_root.join("best_skill.md"), best_skill)?;
    let mut hist_lines = String::new();
    for r in history { hist_lines.push_str(&serde_json::to_string(r)?); hist_lines.push('\n'); }
    std::fs::write(out_root.join("history.jsonl"), hist_lines)?;
    let state = RuntimeState {
        step,
        epoch,
        current_score,
        best_score,
        current_skill_path: cur_path,
        best_skill_path: "best_skill.md".into(),
    };
    std::fs::write(out_root.join("runtime_state.json"), serde_json::to_string_pretty(&state)?)?;
    Ok(())
}
