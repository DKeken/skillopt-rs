use crate::openai::{ChatOptions, OpenAIClient};
use crate::types::{ChatMessage, Edit, EditOp, RejectedEntry, StepBuffer, Trajectory};
use anyhow::Result;
use serde::Deserialize;

const REFLECT_SYS_FALLBACK: &str = "You are SkillOpt, an optimizer that proposes structured edits to a skill document used by a frozen LLM agent.\n\nReturn STRICT JSON: {\"edits\": [...], \"failure_patterns\": [...]}.";

fn render_reflect(kind: &str, has_meta: bool) -> String {
    use minijinja::context;
    crate::templates::env()
        .get_template("reflect")
        .ok()
        .and_then(|t| t.render(context! { kind => kind, has_meta => has_meta, token_limit => 2000 }).ok())
        .unwrap_or_else(|| REFLECT_SYS_FALLBACK.to_string())
}

#[derive(Deserialize)]
struct ReflectOut {
    #[serde(default)]
    edits: Vec<Edit>,
    #[serde(default)]
    failure_patterns: Vec<String>,
}

pub struct ReflectBatch<'a> {
    pub trajectories: &'a [Trajectory],
    pub kind: &'a str, // "success" | "failure"
}

pub async fn reflect(
    client: &OpenAIClient,
    skill: &str,
    batch: &ReflectBatch<'_>,
    buffer: &StepBuffer,
    meta_memo: &str,
    temperature: f32,
    reasoning_effort: &str,
) -> Result<(Vec<Edit>, Vec<String>)> {
    let mut traj_block = String::new();
    for t in batch.trajectories {
        traj_block.push_str(&format!("\n---\nitem: {}\nscore: {:.2}\nprediction: {}\n", t.item_id, t.score, t.prediction));
        for m in &t.messages {
            traj_block.push_str(&format!("[{}] {}\n", m.role, truncate(&m.content, 1200)));
        }
    }
    let buf = format_buffer(buffer);
    let meta_block = if meta_memo.trim().is_empty() {
        String::new()
    } else {
        format!("\n\n=== OPTIMIZER MEMORY (cross-epoch lessons) ===\n{}", meta_memo)
    };
    let user = format!(
        "MINIBATCH KIND: {}\n\n=== CURRENT SKILL ===\n{}\n\n=== TRAJECTORIES ===\n{}\n\n=== BUFFER ===\n{}{}",
        batch.kind, skill, traj_block, buf, meta_block
    );
    let messages = [
        ChatMessage { role: "system".into(), content: render_reflect(batch.kind, !meta_memo.trim().is_empty()) },
        ChatMessage { role: "user".into(), content: user },
    ];
    let out = client.chat(&client.optimizer_model, &messages, ChatOptions {
        temperature: Some(temperature),
        max_tokens: Some(2000),
        json: true,
        reasoning_effort: Some(reasoning_effort),
    }).await?;
    let parsed: ReflectOut = serde_json::from_str(&out.text)
        .or_else(|_| serde_json::from_str::<ReflectOut>(&extract_json(&out.text)))
        .unwrap_or(ReflectOut { edits: vec![], failure_patterns: vec![] });
    let mut edits = parsed.edits;
    for e in &mut edits {
        e.source_type = batch.kind.to_string();
        if e.support_count == 0 { e.support_count = 1; }
    }
    Ok((edits, parsed.failure_patterns))
}

pub fn format_buffer(buffer: &StepBuffer) -> String {
    if buffer.failure_patterns.is_empty() && buffer.rejected.is_empty() {
        return "(empty)".into();
    }
    let mut s = String::new();
    if !buffer.failure_patterns.is_empty() {
        s.push_str("Failure patterns:\n");
        for p in &buffer.failure_patterns {
            s.push_str(&format!("- {}\n", p));
        }
    }
    if !buffer.rejected.is_empty() {
        s.push_str("\nRejected edits (DO NOT REPROPOSE):\n");
        for r in &buffer.rejected {
            s.push_str(&format!("- score_drop {:.3}: {}\n", r.score_drop, r.rationale));
            for e in &r.edits {
                s.push_str(&format!("    [{:?}] anchor=\"{}\" content=\"{}\"\n",
                    e.op, truncate(&e.anchor, 80), truncate(&e.content, 120)));
            }
        }
    }
    s
}

fn extract_json(s: &str) -> String {
    if let (Some(i), Some(j)) = (s.find('{'), s.rfind('}')) {
        if j > i { return s[i..=j].to_string(); }
    }
    "{}".into()
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { return s.to_string(); }
    let mut out: String = s.chars().take(n).collect();
    out.push_str("…");
    out
}

pub fn merge_patches(failure_edits: Vec<Edit>, success_edits: Vec<Edit>) -> Vec<Edit> {
    let mut all = failure_edits;
    all.extend(success_edits);
    let mut out: Vec<Edit> = Vec::with_capacity(all.len());
    for e in all {
        let key = (e.op, e.anchor.clone(), e.content.clone());
        if let Some(existing) = out.iter_mut().find(|x| (x.op, x.anchor.clone(), x.content.clone()) == key) {
            existing.support_count += e.support_count;
            existing.utility = (existing.utility + e.utility) / 2.0;
        } else {
            out.push(e);
        }
    }
    out
}

pub fn rank_and_select(mut edits: Vec<Edit>, lr_budget: u32) -> Vec<Edit> {
    edits.sort_by(|a, b| {
        let sa = a.utility * (a.support_count as f32).sqrt();
        let sb = b.utility * (b.support_count as f32).sqrt();
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    edits.truncate(lr_budget as usize);
    edits
}

pub fn apply_patch(skill: &str, patch: &[Edit]) -> String {
    let mut out = skill.to_string();
    for e in patch {
        match e.op {
            EditOp::Delete => {
                if !e.anchor.is_empty() {
                    if let Some(pos) = out.find(&e.anchor) {
                        let before = &out[..pos];
                        let after_start = pos + e.anchor.len();
                        let after_rest = &out[after_start..];
                        let line_end = after_rest.find('\n').map(|i| after_start + i + 1).unwrap_or(out.len());
                        out = format!("{}{}", before, &out[line_end..]);
                    }
                }
            }
            EditOp::Replace => {
                if !e.anchor.is_empty() {
                    if let Some(pos) = out.find(&e.anchor) {
                        let before = &out[..pos];
                        let after_start = pos + e.anchor.len();
                        let after_rest = &out[after_start..];
                        let line_end = after_rest.find('\n').map(|i| after_start + i).unwrap_or(out.len());
                        out = format!("{}{}{}", before, e.content, &out[line_end..]);
                    } else {
                        out.push_str(&format!("\n{}\n", e.content));
                    }
                }
            }
            EditOp::Add => {
                if e.anchor.is_empty() {
                    out.push_str(&format!("\n{}\n", e.content));
                } else if let Some(pos) = out.find(&e.anchor) {
                    let after_start = pos + e.anchor.len();
                    let after_rest = &out[after_start..];
                    let line_end = after_rest.find('\n').map(|i| after_start + i).unwrap_or(out.len());
                    let before = &out[..line_end];
                    let after = &out[line_end..];
                    out = format!("{}\n{}{}", before, e.content, after);
                } else {
                    out.push_str(&format!("\n{}\n", e.content));
                }
            }
        }
    }
    out
}

pub fn record_rejected(buffer: &mut StepBuffer, edits: Vec<Edit>, drop: f32, rationale: String) {
    buffer.rejected.push(RejectedEntry { edits, score_drop: drop, rationale });
    if buffer.rejected.len() > 12 {
        buffer.rejected.remove(0);
    }
}

const FULL_REWRITE_SYS: &str = "You are SkillOpt's full-rewrite optimizer. Given a current skill document, recent failure trajectories, and the rejected-edit buffer, produce a NEW complete skill document (markdown). Constraints: ≤ 2000 tokens, must address the dominant failure pattern, must preserve any rule that the success-trajectories rely on, must NOT reintroduce any rejected edit verbatim. Output ONLY the new skill markdown — no JSON, no commentary.";

pub async fn full_rewrite(
    client: &OpenAIClient,
    skill: &str,
    failures: &[Trajectory],
    successes: &[Trajectory],
    buffer: &StepBuffer,
    temperature: f32,
    reasoning_effort: &str,
) -> Result<String> {
    let mut block = String::new();
    for (label, traj) in [("FAILURES", failures), ("SUCCESSES", successes)] {
        block.push_str(&format!("\n=== {} ===\n", label));
        for t in traj.iter().take(8) {
            block.push_str(&format!("- item={} score={:.2} pred={}\n", t.item_id, t.score, truncate(&t.prediction, 200)));
        }
    }
    let user = format!(
        "=== CURRENT SKILL ===\n{}\n{}\n=== REJECTED BUFFER ===\n{}\n",
        skill, block, format_buffer(buffer)
    );
    let messages = [
        ChatMessage { role: "system".into(), content: FULL_REWRITE_SYS.into() },
        ChatMessage { role: "user".into(), content: user },
    ];
    let out = client.chat(&client.optimizer_model, &messages, ChatOptions {
        temperature: Some(temperature),
        max_tokens: Some(3000),
        json: false,
        reasoning_effort: Some(reasoning_effort),
    }).await?;
    Ok(out.text.trim().to_string())
}
