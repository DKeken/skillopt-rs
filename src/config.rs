use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub env: EnvCfg,
    pub train: TrainCfg,
    pub gradient: GradientCfg,
    pub optimizer: OptCfg,
    pub evaluation: EvalCfg,
    pub model: ModelCfg,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnvCfg {
    pub name: String,
    pub skill_init: String,
    #[serde(default = "one")]
    pub max_turns: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainCfg {
    pub train_size: usize,
    pub batch_size: usize,
    #[serde(default = "one_usize")]
    pub accumulation: usize,
    pub num_epochs: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GradientCfg {
    pub minibatch_size: usize,
    pub merge_batch_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OptCfg {
    pub learning_rate: u32,
    #[serde(default = "default_schedule")]
    pub lr_schedule: String,
    #[serde(default = "default_true")]
    pub use_rejected_buffer: bool,
    #[serde(default = "default_true")]
    pub use_slow_update: bool,
    #[serde(default = "default_true")]
    pub use_meta_skill: bool,
    #[serde(default)]
    pub full_rewrite_every: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalCfg {
    #[serde(default = "default_sel")]
    pub sel_split: String,
    #[serde(default = "default_test")]
    pub test_split: String,
    #[serde(default = "default_workers")]
    pub workers: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCfg {
    #[serde(default = "default_effort")]
    pub reasoning_effort: String,
    #[serde(default)]
    pub temperature_target: f32,
    #[serde(default = "default_temp_opt")]
    pub temperature_optimizer: f32,
}

fn one() -> u32 { 1 }
fn one_usize() -> usize { 1 }
fn default_schedule() -> String { "cosine".into() }
fn default_true() -> bool { true }
fn default_sel() -> String { "valid_seen".into() }
fn default_test() -> String { "valid_unseen".into() }
fn default_workers() -> usize { 16 }
fn default_effort() -> String { "medium".into() }
fn default_temp_opt() -> f32 { 0.7 }

pub fn load_config(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read config {}", path.display()))?;
    let cfg: Config = serde_yaml::from_str(&text)
        .with_context(|| format!("parse config {}", path.display()))?;
    Ok(cfg)
}
