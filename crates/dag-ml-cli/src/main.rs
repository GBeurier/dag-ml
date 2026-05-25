use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use dag_ml_core::{
    build_execution_plan, oof_campaign_fingerprint, validate_oof_campaign, CampaignSpec,
    ControllerManifest, ControllerRegistry, DagMlError, GraphSpec, OofCampaign,
};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    ValidateGraph {
        path: PathBuf,
    },
    ValidateOofCampaign {
        path: PathBuf,
        #[arg(long)]
        expect_leakage: bool,
    },
    FingerprintOofCampaign {
        path: PathBuf,
    },
    ValidateExecutionPlan {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long, default_value = "plan:cli")]
        plan_id: String,
    },
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
        Command::ValidateOofCampaign {
            path,
            expect_leakage,
        } => {
            let campaign: OofCampaign = read_json(&path, "OOF campaign")?;
            match validate_oof_campaign(&campaign) {
                Ok(matrix) if expect_leakage => {
                    bail!(
                        "expected OOF leakage but campaign joined {} samples and {} columns",
                        matrix.sample_ids.len(),
                        matrix.columns.len()
                    );
                }
                Ok(matrix) => {
                    println!(
                        "valid oof campaign: {} samples, {} columns",
                        matrix.sample_ids.len(),
                        matrix.columns.len()
                    );
                }
                Err(DagMlError::OofLeakage(report)) if expect_leakage => {
                    println!(
                        "expected oof leakage refused at {}: {} violator(s)",
                        report.node_id,
                        report.violators.len()
                    );
                }
                Err(err) => {
                    return Err(err)
                        .with_context(|| format!("invalid OOF campaign at {}", path.display()));
                }
            }
        }
        Command::FingerprintOofCampaign { path } => {
            let campaign: OofCampaign = read_json(&path, "OOF campaign")?;
            let fingerprint = oof_campaign_fingerprint(&campaign)
                .with_context(|| format!("invalid OOF campaign at {}", path.display()))?;
            println!("{fingerprint}");
        }
        Command::ValidateExecutionPlan {
            graph,
            campaign,
            controllers,
            plan_id,
        } => {
            let graph_spec: GraphSpec = read_json(&graph, "graph")?;
            let campaign_spec: CampaignSpec = read_json(&campaign, "campaign")?;
            let controller_manifests: Vec<ControllerManifest> =
                read_json(&controllers, "controller manifest list")?;
            let mut registry = ControllerRegistry::new();
            for manifest in controller_manifests {
                registry.register(manifest)?;
            }
            let plan = build_execution_plan(plan_id, graph_spec, campaign_spec, &registry)
                .with_context(|| "failed to build execution plan")?;
            println!(
                "valid execution plan: {} node(s), {} controller(s), {} variant(s), fold_set={}",
                plan.node_plans.len(),
                plan.controller_manifests.len(),
                plan.variants.len(),
                plan.fold_set
                    .as_ref()
                    .map(|fold_set| fold_set.id.as_str())
                    .unwrap_or("none")
            );
        }
    }

    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf, label: &str) -> Result<T> {
    let data = std::fs::read(path)
        .with_context(|| format!("failed to read {label} JSON at {}", path.display()))?;
    serde_json::from_slice(&data)
        .with_context(|| format!("failed to parse {label} JSON at {}", path.display()))
}
