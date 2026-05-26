use sha2::{Digest, Sha256};

pub fn skill_hash(skill: &str) -> String {
    let mut h = Sha256::new();
    h.update(skill.as_bytes());
    let out = h.finalize();
    let mut s = String::with_capacity(16);
    for b in out.iter().take(8) {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

pub fn normalize_answer(s: &str) -> String {
    let lowered = s.to_lowercase();
    let trimmed: String = lowered.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();
    let collapsed: Vec<&str> = trimmed.split_whitespace().collect();
    let collapsed = collapsed.join(" ");
    let articles = ["a ", "an ", "the "];
    let mut s = collapsed;
    for a in articles {
        if let Some(stripped) = s.strip_prefix(a) {
            s = stripped.to_string();
        }
    }
    s
}

pub fn exact_match(pred: &str, golds: &[String]) -> f32 {
    let p = normalize_answer(pred);
    for g in golds {
        if p == normalize_answer(g) { return 1.0; }
    }
    0.0
}

pub fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f32>() / xs.len() as f32
}
