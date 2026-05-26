use crate::types::{Edit, GateDecision};

pub struct GateInput<'a> {
    pub candidate_score: f32,
    pub current_score: f32,
    pub best_score: f32,
    pub patch: &'a [Edit],
}

pub struct GateOutput {
    pub decision: GateDecision,
    pub score_drop: f32,
}

const EPS: f32 = 1e-4;

pub fn evaluate_gate(input: &GateInput) -> GateOutput {
    let drop = (input.current_score - input.candidate_score).max(0.0);
    if input.candidate_score > input.best_score + EPS {
        GateOutput { decision: GateDecision::AcceptNewBest, score_drop: drop }
    } else if input.candidate_score >= input.current_score - EPS {
        GateOutput { decision: GateDecision::Accept, score_drop: drop }
    } else {
        GateOutput { decision: GateDecision::Reject, score_drop: drop }
    }
}
