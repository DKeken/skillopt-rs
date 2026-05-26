use crate::types::{ChatMessage, Trajectory, TaskItem};
use crate::openai::{OpenAIClient, ChatOptions};
use anyhow::Result;
use futures::stream::{self, StreamExt};

async fn simple_rollout(
    client: &OpenAIClient,
    sys_prompt: String,
    items: &[TaskItem],
    workers: usize,
    temperature: f32,
    max_tokens: u32,
    score_fn: fn(&str, &[String]) -> f32,
) -> Result<Vec<Trajectory>> {
    let model = client.target_model.clone();
    let results: Vec<Result<Trajectory>> = stream::iter(items.iter().cloned())
        .map(|item| {
            let client = client.clone();
            let model = model.clone();
            let sys = sys_prompt.clone();
            async move {
                let user = if item.context.is_empty() {
                    item.question.clone()
                } else {
                    format!("Question: {}\n\n[DOC] {}", item.question, item.context)
                };
                let messages = vec![
                    ChatMessage { role: "system".into(), content: sys },
                    ChatMessage { role: "user".into(), content: user },
                ];
                let out = client.chat(&model, &messages, ChatOptions {
                    temperature: Some(temperature),
                    max_tokens: Some(max_tokens),
                    json: false,
                    reasoning_effort: None,
                }).await?;
                let pred = out.text.trim().to_string();
                let score = score_fn(&pred, &item.answers);
                Ok::<_, anyhow::Error>(Trajectory {
                    item_id: item.id.clone(),
                    prediction: pred,
                    score,
                    messages,
                })
            }
        })
        .buffer_unordered(workers.max(1))
        .collect()
        .await;
    let mut traj = Vec::with_capacity(results.len());
    for r in results { traj.push(r?); }
    Ok(traj)
}

#[async_trait::async_trait]
pub trait Adapter: Send + Sync {
    async fn rollout(
        &self,
        client: &OpenAIClient,
        skill: &str,
        items: &[TaskItem],
        workers: usize,
        temperature: f32,
    ) -> Result<Vec<Trajectory>>;

    fn score(&self, item: &TaskItem, prediction: &str) -> f32;
}

pub struct SearchQAAdapter;

#[async_trait::async_trait]
impl Adapter for SearchQAAdapter {
    async fn rollout(
        &self,
        client: &OpenAIClient,
        skill: &str,
        items: &[TaskItem],
        workers: usize,
        temperature: f32,
    ) -> Result<Vec<Trajectory>> {
        let model = client.target_model.clone();
        let skill = skill.to_string();
        let results: Vec<Result<Trajectory>> = stream::iter(items.iter().cloned())
            .map(|item| {
                let client = client.clone();
                let model = model.clone();
                let skill = skill.clone();
                async move {
                    let messages = vec![
                        ChatMessage {
                            role: "system".into(),
                            content: format!("You are a SearchQA solver. Follow this skill exactly.\n\n---SKILL---\n{}\n---END SKILL---", skill),
                        },
                        ChatMessage {
                            role: "user".into(),
                            content: format!("Question: {}\n\n[DOC] {}", item.question, item.context),
                        },
                    ];
                    let out = client.chat(&model, &messages, ChatOptions {
                        temperature: Some(temperature),
                        max_tokens: Some(200),
                        json: false,
                        reasoning_effort: None,
                    }).await?;
                    let pred = out.text.trim().lines().next().unwrap_or("").trim().to_string();
                    let score = crate::scoring::exact_match(&pred, &item.answers);
                    Ok::<_, anyhow::Error>(Trajectory {
                        item_id: item.id.clone(),
                        prediction: pred,
                        score,
                        messages,
                    })
                }
            })
            .buffer_unordered(workers.max(1))
            .collect()
            .await;
        let mut traj = Vec::with_capacity(results.len());
        for r in results { traj.push(r?); }
        Ok(traj)
    }

    fn score(&self, item: &TaskItem, prediction: &str) -> f32 {
        crate::scoring::exact_match(prediction, &item.answers)
    }
}

pub struct LiveMathAdapter;

#[async_trait::async_trait]
impl Adapter for LiveMathAdapter {
    async fn rollout(
        &self,
        client: &OpenAIClient,
        skill: &str,
        items: &[TaskItem],
        workers: usize,
        temperature: f32,
    ) -> Result<Vec<Trajectory>> {
        let model = client.target_model.clone();
        let skill = skill.to_string();
        let results: Vec<Result<Trajectory>> = stream::iter(items.iter().cloned())
            .map(|item| {
                let client = client.clone();
                let model = model.clone();
                let skill = skill.clone();
                async move {
                    let messages = vec![
                        ChatMessage {
                            role: "system".into(),
                            content: format!("You are a math problem solver. Follow this skill exactly.\n\n---SKILL---\n{}\n---END SKILL---", skill),
                        },
                        ChatMessage {
                            role: "user".into(),
                            content: item.question.clone(),
                        },
                    ];
                    let out = client.chat(&model, &messages, ChatOptions {
                        temperature: Some(temperature),
                        max_tokens: Some(800),
                        json: false,
                        reasoning_effort: None,
                    }).await?;
                    let pred = out.text.trim().to_string();
                    let score = score_numeric(&pred, &item.answers);
                    Ok::<_, anyhow::Error>(Trajectory {
                        item_id: item.id.clone(),
                        prediction: pred,
                        score,
                        messages,
                    })
                }
            })
            .buffer_unordered(workers.max(1))
            .collect()
            .await;
        let mut traj = Vec::with_capacity(results.len());
        for r in results { traj.push(r?); }
        Ok(traj)
    }

    fn score(&self, item: &TaskItem, prediction: &str) -> f32 {
        score_numeric(prediction, &item.answers)
    }
}

fn score_numeric(pred: &str, golds: &[String]) -> f32 {
    let last = extract_last_number(pred);
    for g in golds {
        if let (Some(p), Some(gn)) = (last, parse_number(g)) {
            if (p - gn).abs() < 1e-6 { return 1.0; }
        }
    }
    0.0
}

fn extract_last_number(s: &str) -> Option<f64> {
    let mut last: Option<f64> = None;
    let mut buf = String::new();
    for c in s.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit() || c == '.' || c == '-' {
            buf.push(c);
        } else {
            if !buf.is_empty() {
                if let Ok(n) = buf.parse::<f64>() { last = Some(n); }
                buf.clear();
            }
        }
    }
    last
}

fn parse_number(s: &str) -> Option<f64> {
    s.trim().replace(',', "").parse().ok()
}

// ============== DocVQA ==============
pub struct DocVQAAdapter;

#[async_trait::async_trait]
impl Adapter for DocVQAAdapter {
    async fn rollout(&self, client: &OpenAIClient, skill: &str, items: &[TaskItem], workers: usize, temperature: f32) -> Result<Vec<Trajectory>> {
        let sys = format!("You are a DocVQA solver. Extract answers from the document. Follow this skill exactly.\n\n---SKILL---\n{}\n---END SKILL---", skill);
        simple_rollout(client, sys, items, workers, temperature, 200, score_docvqa).await
    }
    fn score(&self, item: &TaskItem, prediction: &str) -> f32 { score_docvqa(prediction, &item.answers) }
}

fn score_docvqa(pred: &str, golds: &[String]) -> f32 {
    crate::scoring::anls(pred, golds, 0.5)
}

// ============== OfficeQA ==============
pub struct OfficeQAAdapter;

#[async_trait::async_trait]
impl Adapter for OfficeQAAdapter {
    async fn rollout(&self, client: &OpenAIClient, skill: &str, items: &[TaskItem], workers: usize, temperature: f32) -> Result<Vec<Trajectory>> {
        let sys = format!("You answer office-style QA. You may compute inline (dates, arithmetic). Follow this skill exactly. Output the FINAL ANSWER on the last line.\n\n---SKILL---\n{}\n---END SKILL---", skill);
        simple_rollout(client, sys, items, workers, temperature, 600, score_officeqa).await
    }
    fn score(&self, item: &TaskItem, prediction: &str) -> f32 { score_officeqa(prediction, &item.answers) }
}

fn score_officeqa(pred: &str, golds: &[String]) -> f32 {
    let last = pred.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("");
    crate::scoring::exact_match(last, golds).max(score_docvqa(last, golds))
}

// ============== Spreadsheet ==============
pub struct SpreadsheetAdapter;

#[async_trait::async_trait]
impl Adapter for SpreadsheetAdapter {
    async fn rollout(&self, client: &OpenAIClient, skill: &str, items: &[TaskItem], workers: usize, temperature: f32) -> Result<Vec<Trajectory>> {
        let sys = format!("You generate spreadsheet formulas or numeric answers. Output ONLY the final number or formula result. Follow this skill exactly.\n\n---SKILL---\n{}\n---END SKILL---", skill);
        simple_rollout(client, sys, items, workers, temperature, 400, score_numeric).await
    }
    fn score(&self, item: &TaskItem, prediction: &str) -> f32 { score_numeric(prediction, &item.answers) }
}

// ============== ALFWorld-lite ==============
pub struct AlfworldAdapter;

#[async_trait::async_trait]
impl Adapter for AlfworldAdapter {
    async fn rollout(&self, client: &OpenAIClient, skill: &str, items: &[TaskItem], workers: usize, temperature: f32) -> Result<Vec<Trajectory>> {
        let model = client.target_model.clone();
        let skill = skill.to_string();
        let max_turns: usize = 6;
        let results: Vec<Result<Trajectory>> = stream::iter(items.iter().cloned())
            .map(|item| {
                let client = client.clone();
                let model = model.clone();
                let skill = skill.clone();
                async move {
                    let mut messages = vec![
                        ChatMessage { role: "system".into(), content: format!("You are an embodied agent. Output 'THOUGHT: ...\\nACTION: <action>'. End with 'ACTION: done'.\n\n---SKILL---\n{}\n---END SKILL---", skill) },
                        ChatMessage { role: "user".into(), content: format!("Task: {}\nEnvironment: {}", item.question, item.context) },
                    ];
                    let mut final_pred = String::new();
                    for _ in 0..max_turns {
                        let out = client.chat(&model, &messages, ChatOptions {
                            temperature: Some(temperature), max_tokens: Some(300), json: false, reasoning_effort: None,
                        }).await?;
                        messages.push(ChatMessage { role: "assistant".into(), content: out.text.clone() });
                        final_pred = out.text.clone();
                        let lc = out.text.to_lowercase();
                        if lc.contains("action: done") || lc.contains("action:done") { break; }
                        messages.push(ChatMessage { role: "user".into(), content: "OBSERVATION: ok. Continue.".into() });
                    }
                    let score = score_alfworld(&final_pred, &item.answers);
                    Ok::<_, anyhow::Error>(Trajectory { item_id: item.id.clone(), prediction: final_pred, score, messages })
                }
            })
            .buffer_unordered(workers.max(1))
            .collect().await;
        let mut traj = Vec::with_capacity(results.len());
        for r in results { traj.push(r?); }
        Ok(traj)
    }
    fn score(&self, item: &TaskItem, prediction: &str) -> f32 { score_alfworld(prediction, &item.answers) }
}

fn score_alfworld(pred: &str, golds: &[String]) -> f32 {
    let lc = pred.to_lowercase();
    if !lc.contains("done") { return 0.0; }
    for g in golds {
        if lc.contains(&g.to_lowercase()) { return 1.0; }
    }
    0.0
}

// ============== Codex-style harness ==============
pub struct CodexHarnessAdapter {
    pub inner_score_kind: String,
}

#[async_trait::async_trait]
impl Adapter for CodexHarnessAdapter {
    async fn rollout(&self, client: &OpenAIClient, skill: &str, items: &[TaskItem], workers: usize, temperature: f32) -> Result<Vec<Trajectory>> {
        let model = client.target_model.clone();
        let skill = skill.to_string();
        let max_turns: usize = 5;
        let kind = self.inner_score_kind.clone();
        let results: Vec<Result<Trajectory>> = stream::iter(items.iter().cloned())
            .map(|item| {
                let client = client.clone();
                let model = model.clone();
                let skill = skill.clone();
                let kind = kind.clone();
                async move {
                    let mut messages = vec![
                        ChatMessage { role: "system".into(), content: format!("You are a Codex-style agent. You may emit <tool name=\"calc\">EXPR</tool> tags; harness replies with computed value. End with 'FINAL: <answer>' on the last line.\n\n---SKILL---\n{}\n---END SKILL---", skill) },
                        ChatMessage { role: "user".into(), content: format!("Question: {}\n\n[DOC] {}", item.question, item.context) },
                    ];
                    let mut final_pred = String::new();
                    for _ in 0..max_turns {
                        let out = client.chat(&model, &messages, ChatOptions {
                            temperature: Some(temperature), max_tokens: Some(500), json: false, reasoning_effort: None,
                        }).await?;
                        messages.push(ChatMessage { role: "assistant".into(), content: out.text.clone() });
                        if let Some(expr) = extract_calc(&out.text) {
                            let val = eval_calc(&expr).map(|v| v.to_string()).unwrap_or_else(|| "ERR".into());
                            messages.push(ChatMessage { role: "user".into(), content: format!("<observation>{}</observation>", val) });
                            continue;
                        }
                        final_pred = extract_final(&out.text).unwrap_or(out.text.clone());
                        break;
                    }
                    let score = match kind.as_str() {
                        "numeric" => score_numeric(&final_pred, &item.answers),
                        _ => crate::scoring::exact_match(&final_pred, &item.answers),
                    };
                    Ok::<_, anyhow::Error>(Trajectory { item_id: item.id.clone(), prediction: final_pred, score, messages })
                }
            })
            .buffer_unordered(workers.max(1))
            .collect().await;
        let mut traj = Vec::with_capacity(results.len());
        for r in results { traj.push(r?); }
        Ok(traj)
    }
    fn score(&self, item: &TaskItem, prediction: &str) -> f32 {
        match self.inner_score_kind.as_str() {
            "numeric" => score_numeric(prediction, &item.answers),
            _ => crate::scoring::exact_match(prediction, &item.answers),
        }
    }
}

pub fn extract_calc(s: &str) -> Option<String> {
    let i = s.find("<tool name=\"calc\">")?;
    let rest = &s[i + 18..];
    let j = rest.find("</tool>")?;
    Some(rest[..j].trim().to_string())
}

pub fn eval_calc(expr: &str) -> Option<f64> {
    use evalexpr::Value;
    let promoted = expr.replace('/', "*1.0/");
    let try_float = evalexpr::eval(&promoted);
    let result = if try_float.is_ok() { try_float } else { evalexpr::eval(expr) };
    let n = match result {
        Ok(Value::Float(f)) => f,
        Ok(Value::Int(i)) => i as f64,
        _ => return None,
    };
    if n.is_finite() { Some(n) } else { None }
}

pub fn extract_final(s: &str) -> Option<String> {
    s.lines().rev().find_map(|l| {
        let t = l.trim();
        t.strip_prefix("FINAL:").or_else(|| t.strip_prefix("Final:")).map(|x| x.trim().to_string())
    })
}

pub fn build_adapter(name: &str) -> Box<dyn Adapter> {
    match name {
        "searchqa" => Box::new(SearchQAAdapter),
        "livemath" => Box::new(LiveMathAdapter),
        "docvqa" => Box::new(DocVQAAdapter),
        "officeqa" => Box::new(OfficeQAAdapter),
        "spreadsheet" => Box::new(SpreadsheetAdapter),
        "alfworld" => Box::new(AlfworldAdapter),
        "codex_searchqa" => Box::new(CodexHarnessAdapter { inner_score_kind: "exact".into() }),
        "codex_livemath" => Box::new(CodexHarnessAdapter { inner_score_kind: "numeric".into() }),
        other => panic!("unknown adapter: {}", other),
    }
}

#[derive(Clone)]
pub struct OpenAIClientCloneable;
