use std::f32::consts::PI;

pub enum LrSchedule { Constant, Linear, Cosine, Autonomous }

impl LrSchedule {
    pub fn from_str(s: &str) -> Self {
        match s {
            "constant" => Self::Constant,
            "linear" => Self::Linear,
            "autonomous" => Self::Autonomous,
            _ => Self::Cosine,
        }
    }
}

/// Compute edit budget L_t. base_lr is the configured top-end lr.
/// For Autonomous, returns base_lr as upper bound — caller should use
/// `autonomous_select` instead of compute_lr+rank_and_select.
pub fn compute_lr(sched: &LrSchedule, step: u32, total_steps: u32, base_lr: u32) -> u32 {
    if total_steps == 0 { return base_lr.max(1); }
    let progress = (step as f32 / total_steps.max(1) as f32).min(1.0);
    let min_lr = 1.0_f32;
    let base = base_lr.max(1) as f32;
    let lr = match sched {
        LrSchedule::Constant | LrSchedule::Autonomous => base,
        LrSchedule::Linear => (base - (base - min_lr) * progress).max(min_lr),
        LrSchedule::Cosine => {
            min_lr + 0.5 * (base - min_lr) * (1.0 + (PI * progress).cos())
        }
    };
    lr.round().max(1.0) as u32
}

/// Autonomous selection: take edits in ranked order while cumulative
/// confidence stays under `confidence_budget` (default 1.5). Hard upper
/// bound = `cap` to avoid pathological large patches.
pub fn autonomous_select(edits: Vec<crate::types::Edit>, confidence_budget: f32, cap: u32) -> Vec<crate::types::Edit> {
    let mut sorted = edits;
    sorted.sort_by(|a, b| {
        let sa = a.utility * (a.support_count as f32).sqrt();
        let sb = b.utility * (b.support_count as f32).sqrt();
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut acc = 0.0f32;
    let mut out = Vec::new();
    for e in sorted {
        if out.len() as u32 >= cap { break; }
        let conf = e.utility.clamp(0.0, 1.0);
        if acc + conf > confidence_budget && !out.is_empty() { break; }
        acc += conf;
        out.push(e);
    }
    out
}
