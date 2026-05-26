use std::f32::consts::PI;

pub enum LrSchedule { Constant, Linear, Cosine }

impl LrSchedule {
    pub fn from_str(s: &str) -> Self {
        match s {
            "constant" => Self::Constant,
            "linear" => Self::Linear,
            _ => Self::Cosine,
        }
    }
}

/// Compute edit budget L_t. base_lr is the configured top-end lr.
pub fn compute_lr(sched: &LrSchedule, step: u32, total_steps: u32, base_lr: u32) -> u32 {
    if total_steps == 0 { return base_lr.max(1); }
    let progress = (step as f32 / total_steps.max(1) as f32).min(1.0);
    let min_lr = 1.0_f32;
    let base = base_lr.max(1) as f32;
    let lr = match sched {
        LrSchedule::Constant => base,
        LrSchedule::Linear => (base - (base - min_lr) * progress).max(min_lr),
        LrSchedule::Cosine => {
            min_lr + 0.5 * (base - min_lr) * (1.0 + (PI * progress).cos())
        }
    };
    lr.round().max(1.0) as u32
}
