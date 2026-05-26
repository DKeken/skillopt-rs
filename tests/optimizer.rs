use skillopt::adapter::{eval_calc, extract_calc, extract_final};
use skillopt::gate::{evaluate_gate, GateInput};
use skillopt::gradient::{apply_patch, merge_patches, rank_and_select, record_rejected};
use skillopt::memory::replace_slow_update_field;
use skillopt::scheduler::{autonomous_select, compute_lr, LrSchedule};
use skillopt::scoring::{anls, count_tokens, exact_match, normalize_answer, skill_hash};
use skillopt::types::{Edit, EditOp, GateDecision, StepBuffer};

fn e(op: EditOp, anchor: &str, content: &str, utility: f32, support: u32, source: &str) -> Edit {
    Edit {
        op,
        anchor: anchor.into(),
        content: content.into(),
        rationale: "".into(),
        utility,
        source_type: source.into(),
        support_count: support,
    }
}

#[test]
fn cosine_lr_decays_from_base_to_one() {
    let s = LrSchedule::Cosine;
    assert_eq!(compute_lr(&s, 0, 100, 4), 4);
    assert!(compute_lr(&s, 50, 100, 4) <= 4);
    assert_eq!(compute_lr(&s, 100, 100, 4), 1);
}

#[test]
fn linear_lr_monotone_decreasing() {
    let s = LrSchedule::Linear;
    let a = compute_lr(&s, 0, 100, 8);
    let b = compute_lr(&s, 50, 100, 8);
    let c = compute_lr(&s, 100, 100, 8);
    assert!(a >= b && b >= c);
    assert_eq!(c, 1);
}

#[test]
fn constant_lr_unchanged() {
    let s = LrSchedule::Constant;
    assert_eq!(compute_lr(&s, 0, 100, 4), 4);
    assert_eq!(compute_lr(&s, 50, 100, 4), 4);
    assert_eq!(compute_lr(&s, 100, 100, 4), 4);
}

#[test]
fn lr_never_below_one() {
    for sched in [LrSchedule::Cosine, LrSchedule::Linear, LrSchedule::Constant] {
        for step in 0..120 {
            let lr = compute_lr(&sched, step, 100, 4);
            assert!(lr >= 1, "step {} sched gave {}", step, lr);
        }
    }
}

#[test]
fn gate_accepts_strict_improvement_as_new_best() {
    let patch: Vec<Edit> = vec![];
    let g = evaluate_gate(&GateInput {
        candidate_score: 0.92,
        current_score: 0.85,
        best_score: 0.85,
        patch: &patch,
    });
    assert!(matches!(g.decision, GateDecision::AcceptNewBest));
    assert!(g.score_drop < 1e-6);
}

#[test]
fn gate_accepts_equal_to_current_no_new_best() {
    let patch: Vec<Edit> = vec![];
    let g = evaluate_gate(&GateInput {
        candidate_score: 0.85,
        current_score: 0.85,
        best_score: 0.90,
        patch: &patch,
    });
    assert!(matches!(g.decision, GateDecision::Accept));
}

#[test]
fn gate_rejects_regression_and_reports_drop() {
    let patch: Vec<Edit> = vec![];
    let g = evaluate_gate(&GateInput {
        candidate_score: 0.70,
        current_score: 0.85,
        best_score: 0.90,
        patch: &patch,
    });
    assert!(matches!(g.decision, GateDecision::Reject));
    assert!((g.score_drop - 0.15).abs() < 1e-4);
}

#[test]
fn rank_and_select_clips_to_budget_by_utility_and_support() {
    let edits = vec![
        e(EditOp::Add, "", "low", 0.1, 1, "failure"),
        e(EditOp::Add, "", "mid", 0.5, 4, "failure"),
        e(EditOp::Add, "", "high", 0.9, 9, "failure"),
        e(EditOp::Add, "", "tied", 0.9, 1, "failure"),
    ];
    let picked = rank_and_select(edits, 2);
    assert_eq!(picked.len(), 2);
    assert_eq!(picked[0].content, "high");
}

#[test]
fn merge_patches_dedups_and_aggregates_support() {
    let a = vec![e(EditOp::Add, "anchor", "x", 0.5, 1, "failure")];
    let b = vec![
        e(EditOp::Add, "anchor", "x", 0.7, 1, "success"),
        e(EditOp::Add, "anchor", "y", 0.4, 1, "success"),
    ];
    let merged = merge_patches(a, b);
    assert_eq!(merged.len(), 2);
    let dup = merged.iter().find(|m| m.content == "x").unwrap();
    assert_eq!(dup.support_count, 2);
    assert!((dup.utility - 0.6).abs() < 1e-4);
}

#[test]
fn apply_patch_add_at_end_when_anchor_empty() {
    let skill = "# title\n\nrule one\n";
    let patch = vec![e(EditOp::Add, "", "rule two", 0.5, 1, "failure")];
    let out = apply_patch(skill, &patch);
    assert!(out.contains("rule one"));
    assert!(out.contains("rule two"));
    assert!(out.find("rule two").unwrap() > out.find("rule one").unwrap());
}

#[test]
fn apply_patch_add_after_anchor_line() {
    let skill = "# title\n\nrule one\nrule three\n";
    let patch = vec![e(EditOp::Add, "rule one", "rule two", 0.5, 1, "failure")];
    let out = apply_patch(skill, &patch);
    let i1 = out.find("rule one").unwrap();
    let i2 = out.find("rule two").unwrap();
    let i3 = out.find("rule three").unwrap();
    assert!(i1 < i2 && i2 < i3);
}

#[test]
fn apply_patch_replace_swaps_line() {
    let skill = "alpha\nold rule\nbeta\n";
    let patch = vec![e(EditOp::Replace, "old rule", "new rule", 0.5, 1, "failure")];
    let out = apply_patch(skill, &patch);
    assert!(!out.contains("old rule"));
    assert!(out.contains("new rule"));
    assert!(out.contains("alpha") && out.contains("beta"));
}

#[test]
fn apply_patch_delete_removes_line() {
    let skill = "alpha\nremove me\nbeta\n";
    let patch = vec![e(EditOp::Delete, "remove me", "", 0.5, 1, "failure")];
    let out = apply_patch(skill, &patch);
    assert!(!out.contains("remove me"));
    assert!(out.contains("alpha"));
    assert!(out.contains("beta"));
}

#[test]
fn rejected_buffer_caps_at_twelve_and_drops_oldest() {
    let mut buf = StepBuffer::default();
    for i in 0..20 {
        let edits = vec![e(EditOp::Add, "", &format!("e{}", i), 0.5, 1, "failure")];
        record_rejected(&mut buf, edits, 0.05, format!("r{}", i));
    }
    assert_eq!(buf.rejected.len(), 12);
    assert!(buf.rejected.first().unwrap().rationale.starts_with("r"));
    assert_eq!(buf.rejected.last().unwrap().rationale, "r19");
}

#[test]
fn slow_update_replaces_existing_section() {
    let skill = "# main\n\nrule\n\n## Slow update\nold guidance\n\n## footer\nz\n";
    let out = replace_slow_update_field(skill, "fresh guidance");
    assert!(out.contains("fresh guidance"));
    assert!(!out.contains("old guidance"));
    assert!(out.contains("## footer"));
}

#[test]
fn slow_update_appends_when_missing() {
    let skill = "# main\n\nrule\n";
    let out = replace_slow_update_field(skill, "fresh guidance");
    assert!(out.contains("## Slow update"));
    assert!(out.contains("fresh guidance"));
}

#[test]
fn normalize_answer_handles_articles_and_punct() {
    assert_eq!(normalize_answer("The U.S.A."), "usa");
    assert_eq!(normalize_answer("a   Cat!"), "cat");
    assert_eq!(normalize_answer("An orange."), "orange");
}

#[test]
fn exact_match_compares_normalized_forms() {
    let golds = vec!["The Beatles".to_string(), "beatles".into()];
    assert_eq!(exact_match("the beatles", &golds), 1.0);
    assert_eq!(exact_match("BEATLES", &golds), 1.0);
    assert_eq!(exact_match("rolling stones", &golds), 0.0);
}

#[test]
fn skill_hash_is_deterministic_and_short() {
    let a = skill_hash("hello world");
    let b = skill_hash("hello world");
    let c = skill_hash("hello worlds");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(a.len(), 16);
}

#[test]
fn autonomous_select_respects_confidence_budget() {
    let edits = vec![
        e(EditOp::Add, "", "a", 0.9, 4, "failure"),
        e(EditOp::Add, "", "b", 0.4, 1, "failure"),
        e(EditOp::Add, "", "c", 0.3, 1, "failure"),
        e(EditOp::Add, "", "d", 0.1, 1, "failure"),
    ];
    let picked = autonomous_select(edits, 1.5, 8);
    assert!(picked.len() >= 1 && picked.len() <= 4);
    assert_eq!(picked[0].content, "a");
}

#[test]
fn autonomous_select_respects_cap() {
    let edits = (0..50).map(|i| e(EditOp::Add, "", &format!("e{}", i), 0.01, 1, "failure")).collect();
    let picked = autonomous_select(edits, 100.0, 4);
    assert_eq!(picked.len(), 4);
}

#[test]
fn lr_schedule_autonomous_falls_back_to_constant_for_compute_lr() {
    let s = LrSchedule::Autonomous;
    assert_eq!(compute_lr(&s, 0, 100, 4), 4);
    assert_eq!(compute_lr(&s, 50, 100, 4), 4);
    assert_eq!(compute_lr(&s, 100, 100, 4), 4);
}

#[test]
fn extract_calc_pulls_expression_from_tool_tag() {
    let s = "I will compute <tool name=\"calc\">12 + 30</tool> next";
    assert_eq!(extract_calc(s), Some("12 + 30".to_string()));
}

#[test]
fn extract_calc_returns_none_when_absent() {
    assert_eq!(extract_calc("just text"), None);
}

#[test]
fn eval_calc_handles_basic_arithmetic_and_precedence() {
    assert_eq!(eval_calc("2 + 3"), Some(5.0));
    assert_eq!(eval_calc("2 + 3 * 4"), Some(14.0));
    assert_eq!(eval_calc("(2 + 3) * 4"), Some(20.0));
    assert_eq!(eval_calc("10 / 4"), Some(2.5));
    assert_eq!(eval_calc("10 / 0"), None);
}

#[test]
fn extract_final_picks_last_final_line() {
    let s = "thinking...\nFINAL: 42\n";
    assert_eq!(extract_final(s).as_deref(), Some("42"));
    assert_eq!(extract_final("Final: forty\n").as_deref(), Some("forty"));
    assert_eq!(extract_final("no marker"), None);
}

#[test]
fn anls_perfect_match_is_one() {
    let golds = vec!["Coca Cola".to_string(), "Coca Cola Company".into()];
    assert!((anls("Coca Cola", &golds, 0.5) - 1.0).abs() < 1e-4);
}

#[test]
fn anls_close_match_above_threshold() {
    let golds = vec!["Coca Cola".to_string()];
    let s = anls("CocaCola", &golds, 0.5);
    assert!(s > 0.5 && s < 1.0, "got {}", s);
}

#[test]
fn anls_far_mismatch_returns_zero() {
    let golds = vec!["Coca Cola".to_string()];
    assert_eq!(anls("Pepsi", &golds, 0.5), 0.0);
}

#[test]
fn anls_empty_inputs_zero() {
    let golds = vec!["x".to_string()];
    assert_eq!(anls("", &golds, 0.5), 0.0);
    assert_eq!(anls("x", &[], 0.5), 0.0);
}

#[test]
fn count_tokens_returns_nonzero_for_text() {
    let n = count_tokens("hello world this is a sentence");
    assert!(n > 0 && n < 20, "got {}", n);
}

#[test]
fn count_tokens_grows_with_length() {
    let a = count_tokens("short");
    let b = count_tokens(&"long ".repeat(200));
    assert!(b > a * 50, "a={} b={}", a, b);
}

#[test]
fn eval_calc_powers_via_evalexpr() {
    assert_eq!(eval_calc("2 ^ 8"), Some(256.0));
    assert_eq!(eval_calc("3 ^ 4"), Some(81.0));
}
