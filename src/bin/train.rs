use anyhow::{Context, Result};
use clap::Parser;
use skillopt::adapter::{build_adapter, Adapter};
use skillopt::config::load_config;
use skillopt::data::{load_split, sample_batch};
use skillopt::gate::{evaluate_gate, GateInput};
use skillopt::gradient::{
    apply_patch, merge_patches, rank_and_select, record_rejected, reflect, ReflectBatch,
};
use skillopt::memory::{replace_slow_update_field, run_meta_skill, run_slow_update};
use skillopt::openai::OpenAIClient;
use skillopt::scheduler::{compute_lr, LrSchedule};
use skillopt::scoring::{mean, skill_hash};
use skillopt::types::{Edit, GateDecision, StepBuffer, StepRecord, Trajectory};
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

    let mut skill = std::fs::read_to_string(&cfg.env.skill_init)
        .with_context(|| format!("read skill_init {}", cfg.env.skill_init))?;
    let mut best_skill = skill.clone();
    let _ = std::fs::write(cli.out_root.join("skills/skill_v0000.md"), &skill);

    let baseline = adapter
        .rollout(&client, &skill, &sel_items, cfg.evaluation.workers, cfg.model.temperature_target)
        .await?;
    let mut current_score = mean(&baseline.iter().map(|t| t.score).collect::<Vec<_>>());
    let mut best_score = current_score;
    info!("baseline sel_score = {:.3}", baseline_pretty(current_score));

    let sched = LrSchedule::from_str(&cfg.optimizer.lr_schedule);
    let steps_per_epoch = ((cfg.train.train_size as f32) / (cfg.train.batch_size * cfg.train.accumulation) as f32).ceil() as u32;
    let total_steps = cfg.train.num_epochs * steps_per_epoch.max(1);

    let mut history: Vec<StepRecord> = vec![];
    let mut step_idx: u32 = 0;

    let mut prev_epoch_skill = skill.clone();

    for epoch in 1..=cfg.train.num_epochs {
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
                    cfg.model.temperature_optimizer,
                    &cfg.model.reasoning_effort,
                ).await?;
                all_success_edits.extend(edits);
            }

            // 3. Aggregate
            let merged = merge_patches(all_failure_edits, all_success_edits);
            let n_proposed = merged.len() as u32;

            // 4. Select under edit budget
            let lr = compute_lr(&sched, step_idx - 1, total_steps, cfg.optimizer.learning_rate);
            let patch = rank_and_select(merged, lr);
            let n_selected = patch.len() as u32;

            // 5. Update
            let candidate = apply_patch(&skill, &patch);
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
            persist_step(&cli.out_root, step_idx, &skill, &best_skill, &rec, &history)?;
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
            std::fs::write(cli.out_root.join(format!("meta_skill/epoch_{epoch:02}/memo.md")), memo)?;
        }
        prev_epoch_skill = skill.clone();
    }

    std::fs::write(cli.out_root.join("best_skill.md"), &best_skill)?;
    println!("\nbest_sel_score = {:.3}", best_score);
    println!("artifact: {}", cli.out_root.join("best_skill.md").display());
    Ok(())
}

fn baseline_pretty(x: f32) -> f32 { x }

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
    skill: &str,
    best_skill: &str,
    rec: &StepRecord,
    history: &[StepRecord],
) -> Result<()> {
    let dir = out_root.join(format!("steps/step_{:04}", step));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("record.json"), serde_json::to_string_pretty(rec)?)?;
    std::fs::write(out_root.join(format!("skills/skill_v{:04}.md", step)), skill)?;
    std::fs::write(out_root.join("best_skill.md"), best_skill)?;
    let mut hist_lines = String::new();
    for r in history { hist_lines.push_str(&serde_json::to_string(r)?); hist_lines.push('\n'); }
    std::fs::write(out_root.join("history.jsonl"), hist_lines)?;
    let state = serde_json::json!({ "step": step, "best_sel_score": rec.best_sel_score });
    std::fs::write(out_root.join("runtime_state.json"), serde_json::to_string_pretty(&state)?)?;
    Ok(())
}
