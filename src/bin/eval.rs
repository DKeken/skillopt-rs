use anyhow::Result;
use clap::Parser;
use skillopt::adapter::build_adapter;
use skillopt::config::load_config;
use skillopt::data::load_split;
use skillopt::openai::OpenAIClient;
use skillopt::scoring::mean;
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    #[arg(long)] config: PathBuf,
    #[arg(long)] skill: PathBuf,
    #[arg(long, default_value = "valid_unseen")] split: String,
    #[arg(long)] split_dir: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("skillopt=info,info").init();

    let cli = Cli::parse();
    let cfg = load_config(&cli.config)?;
    let client = OpenAIClient::from_env()?;
    let adapter = build_adapter(&cfg.env.name);

    let skill = std::fs::read_to_string(&cli.skill)?;
    let items = load_split(&cli.split_dir, &cli.split)?;

    let traj = adapter.rollout(&client, &skill, &items, cfg.evaluation.workers, cfg.model.temperature_target).await?;
    let score = mean(&traj.iter().map(|t| t.score).collect::<Vec<_>>());

    println!("split = {}  n = {}  score = {:.3}", cli.split, items.len(), score);
    Ok(())
}
