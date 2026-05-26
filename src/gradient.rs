use crate::openai::{ChatOptions, OpenAIClient};
use crate::types::{ChatMessage, Edit, EditOp, RejectedEntry, StepBuffer, Trajectory};
use anyhow::Result;
use serde::Deserialize;

const REFLECT_SYS: &str = "You are SkillOpt, an optimizer that proposes structured edits to a skill document used by a frozen LLM agent.

You will receive:
- The current skill document.
- A minibatch of trajectories (success or failure) the target produced.
- A buffer of failure patterns and rejected past edits with the score drop they caused.

Your job: propose between 0 and 8 structured edits to the skill that will help on the FAILURES without breaking the SUCCESSES. Edits must be small and reusable, never task-specific.

Return STRICT JSON:
{
  \"edits\": [
    {
      \"op\": \"add\" | \"delete\" | \"replace\",
      \"anchor\": \"verbatim line/heading from current skill (empty for plain add at end)\",
      \"content\": \"new bullet/section to insert OR replacement text (ignored for delete)\",
      \"rationale\": \"why this helps the failure pattern\",
      \"utility\": 0.0 to 1.0
    }
  ],
  \"failure_patterns\": [\"short phrases naming recurring failure modes\"]
}

Guidelines:
- Prefer add/replace over delete unless a rule clearly hurts.
- Keep skill under 2000 tokens; prune dead weight when adding.
- Do not repeat any rejected edit from the buffer.
- If no useful edits, return {\"edits\": [], \"failure_patterns\": [...]}.
";

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
    let user = format!(
        "MINIBATCH KIND: {}\n\n=== CURRENT SKILL ===\n{}\n\n=== TRAJECTORIES ===\n{}\n\n=== BUFFER ===\n{}",
        batch.kind, skill, traj_block, buf
    );
    let messages = [
        ChatMessage { role: "system".into(), content: REFLECT_SYS.into() },
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
