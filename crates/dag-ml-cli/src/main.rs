use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dag_ml_core::GraphSpec;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    ValidateGraph { path: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::ValidateGraph { path } => {
            let data = std::fs::read(&path)
                .with_context(|| format!("failed to read graph JSON at {}", path.display()))?;
            let graph: GraphSpec = serde_json::from_slice(&data)
                .with_context(|| format!("failed to parse graph JSON at {}", path.display()))?;
            graph
                .validate()
                .with_context(|| format!("invalid graph at {}", path.display()))?;
            println!("valid graph: {}", graph.id);
        }
    }

    Ok(())
}
