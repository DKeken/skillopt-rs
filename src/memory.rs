use crate::openai::{ChatOptions, OpenAIClient};
use crate::types::{ChatMessage, Trajectory};
use anyhow::Result;

const META_SYS: &str = "You are SkillOpt's meta-skill writer. Read longitudinal pairs (same task solved with previous-epoch and current-epoch skills) and produce a SHORT, optimizer-side memo (<= 30 lines) summarizing durable lessons across epochs. This memo is NOT shipped with the deployed skill; it is consumed by the optimizer in the next epoch.";

const SLOW_SYS: &str = "You are SkillOpt's slow-update writer. From longitudinal comparison pairs, produce a single block of guidance (<= 200 tokens) to be force-injected into the skill under the heading '## Slow update' — only durable, cross-epoch rules.";

pub async fn run_meta_skill(
    client: &OpenAIClient,
    pairs: &[(Trajectory, Trajectory)],
    temperature: f32,
    reasoning_effort: &str,
) -> Result<String> {
    if pairs.is_empty() { return Ok(String::new()); }
    let mut block = String::new();
    for (prev, curr) in pairs.iter().take(20) {
        block.push_str(&format!(
            "\n--\nitem: {}\nprev_score: {:.2}\ncurr_score: {:.2}\nprev_pred: {}\ncurr_pred: {}\n",
            prev.item_id, prev.score, curr.score, prev.prediction, curr.prediction
        ));
    }
    let messages = [
        ChatMessage { role: "system".into(), content: META_SYS.into() },
        ChatMessage { role: "user".into(), content: block },
    ];
    let out = client.chat(&client.optimizer_model, &messages, ChatOptions {
        temperature: Some(temperature),
        max_tokens: Some(800),
        json: false,
        reasoning_effort: Some(reasoning_effort),
    }).await?;
    Ok(out.text)
}

pub async fn run_slow_update(
    client: &OpenAIClient,
    pairs: &[(Trajectory, Trajectory)],
    temperature: f32,
    reasoning_effort: &str,
) -> Result<String> {
    if pairs.is_empty() { return Ok(String::new()); }
    let mut block = String::new();
    for (prev, curr) in pairs.iter().take(20) {
        block.push_str(&format!(
            "\n--\nitem: {}\nprev_score: {:.2}\ncurr_score: {:.2}\n",
            prev.item_id, prev.score, curr.score
        ));
    }
    let messages = [
        ChatMessage { role: "system".into(), content: SLOW_SYS.into() },
        ChatMessage { role: "user".into(), content: block },
    ];
    let out = client.chat(&client.optimizer_model, &messages, ChatOptions {
        temperature: Some(temperature),
        max_tokens: Some(400),
        json: false,
        reasoning_effort: Some(reasoning_effort),
    }).await?;
    Ok(out.text)
}

pub fn replace_slow_update_field(skill: &str, block: &str) -> String {
    let header = "## Slow update";
    let new_section = format!("{}\n{}\n", header, block.trim());
    if let Some(start) = skill.find(header) {
        let end = skill[start..].find("\n## ").map(|i| start + i).unwrap_or(skill.len());
        let mut out = String::new();
        out.push_str(&skill[..start]);
        out.push_str(&new_section);
        out.push_str(&skill[end..]);
        out
    } else {
        let mut out = skill.to_string();
        if !out.ends_with('\n') { out.push('\n'); }
        out.push_str(&new_section);
        out
    }
}
