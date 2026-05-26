use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use std::path::Path;

use crate::types::TaskItem;

pub fn load_split(split_dir: &Path, split: &str) -> Result<Vec<TaskItem>> {
    let path = split_dir.join(split).join("items.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read split {}", path.display()))?;
    let items: Vec<TaskItem> = serde_json::from_str(&text)
        .with_context(|| format!("parse split {}", path.display()))?;
    Ok(items)
}

pub fn sample_batch(items: &[TaskItem], n: usize, seed: u64) -> Vec<TaskItem> {
    if items.len() <= n {
        return items.to_vec();
    }
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..items.len()).collect();
    idx.shuffle(&mut rng);
    idx.into_iter().take(n).map(|i| items[i].clone()).collect()
}
