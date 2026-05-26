use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskItem {
    pub id: String,
    pub question: String,
    #[serde(default)]
    pub context: String,
    pub answers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub item_id: String,
    pub prediction: String,
    pub score: f32,
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EditOp {
    Add,
    Delete,
    Replace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edit {
    pub op: EditOp,
    pub anchor: String,
    pub content: String,
    #[serde(default)]
    pub rationale: String,
    #[serde(default = "default_utility")]
    pub utility: f32,
    #[serde(default)]
    pub source_type: String,
    #[serde(default = "default_support")]
    pub support_count: u32,
}

fn default_utility() -> f32 { 0.5 }
fn default_support() -> u32 { 1 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedEntry {
    pub edits: Vec<Edit>,
    pub score_drop: f32,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StepBuffer {
    pub failure_patterns: Vec<String>,
    pub rejected: Vec<RejectedEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateDecision {
    AcceptNewBest,
    Accept,
    Reject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub step: u32,
    pub epoch: u32,
    pub lr_budget: u32,
    pub train_score: f32,
    pub sel_score: f32,
    pub best_sel_score: f32,
    pub gate: GateDecision,
    pub n_proposed: u32,
    pub n_selected: u32,
    pub skill_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeState {
    pub step: u32,
    pub epoch: u32,
    pub current_score: f32,
    pub best_score: f32,
    pub current_skill_path: String,
    pub best_skill_path: String,
}
