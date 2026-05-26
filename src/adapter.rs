use crate::types::{ChatMessage, Trajectory, TaskItem};
use crate::openai::{OpenAIClient, ChatOptions};
use anyhow::Result;
use futures::stream::{self, StreamExt};

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

pub fn build_adapter(name: &str) -> Box<dyn Adapter> {
    match name {
        "searchqa" => Box::new(SearchQAAdapter),
        other => panic!("unknown adapter: {}", other),
    }
}

#[derive(Clone)]
pub struct OpenAIClientCloneable;
