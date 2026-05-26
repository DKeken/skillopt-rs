use minijinja::Environment;
use std::sync::OnceLock;

static ENV: OnceLock<Environment<'static>> = OnceLock::new();

pub fn env() -> &'static Environment<'static> {
    ENV.get_or_init(|| {
        let mut env = Environment::new();
        env.add_template("reflect", include_str!("prompts/reflect.j2")).unwrap();
        env.add_template("full_rewrite", include_str!("prompts/full_rewrite.j2")).unwrap();
        env.add_template("meta_skill", include_str!("prompts/meta_skill.j2")).unwrap();
        env.add_template("slow_update", include_str!("prompts/slow_update.j2")).unwrap();
        env
    })
}
