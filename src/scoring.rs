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

/// ANLS = Average Normalized Levenshtein Similarity (Biten+ ICCV'19).
/// Standard DocVQA metric. Score = max over gold answers of:
///   1 - NL(pred, gold)  if NL < threshold,
///   0                   otherwise.
/// Default threshold τ = 0.5 per the paper.
pub fn anls(pred: &str, golds: &[String], threshold: f32) -> f32 {
    if pred.is_empty() || golds.is_empty() { return 0.0; }
    let p = pred.to_lowercase();
    golds.iter().map(|g| {
        let g = g.to_lowercase();
        let nl = strsim::normalized_levenshtein(&p, &g) as f32; // similarity, not distance
        let nl_dist = 1.0 - nl;
        if nl_dist < threshold { 1.0 - nl_dist } else { 0.0 }
    }).fold(0.0_f32, f32::max)
}

/// Soft token count using tiktoken-rs (cl100k_base — fine for guard purposes).
/// Cached singleton; falls back to char/4 heuristic if init fails.
pub fn count_tokens(text: &str) -> usize {
    use std::sync::OnceLock;
    static BPE: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();
    let bpe = BPE.get_or_init(|| tiktoken_rs::cl100k_base().ok());
    if let Some(b) = bpe {
        b.encode_with_special_tokens(text).len()
    } else {
        text.len() / 4
    }
}
