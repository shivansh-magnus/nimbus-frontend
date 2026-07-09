"""
Day-10 Nimbus CLI — a single installable Typer entrypoint replacing the
scattered argparse scripts.

Install via pyproject.toml [project.scripts]:
    nimbus = "automl_agents.cli:app"

Usage:
    uv run nimbus --help
    uv run nimbus run --csv data/raw/synthetic_ground_truth.csv --target churn
    uv run nimbus stress-test
    uv run nimbus verify-providers
    uv run nimbus generate-data
    uv run nimbus download-data
    uv run nimbus serve-mcp
    uv run nimbus serve-mcp --transport stdio

Design (from DAY10_NEXT_STEPS.md §2):
  - Every command is a thin wrapper calling the corresponding extracted
    script function — no subprocess shelling.
  - The scripts themselves remain runnable exactly as before.
  - serve-mcp is a subcommand of this same binary (not a separate console
    script) — one entrypoint, one --help tree to discover everything.
  - rich.Console is used for formatted output; raw print() is used inside
    the underlying functions so both CLI and direct invocation are readable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure src/ is importable when running via `uv run nimbus`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# Also ensure scripts/ is importable (for the script functions we delegate to)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = typer.Typer(
    name="nimbus",
    help="[bold cyan]Nimbus[/bold cyan] - Multi-agent AutoML pipeline CLI.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    csv: str = typer.Option(
        "data/raw/synthetic_ground_truth.csv",
        "--csv",
        help="Path to raw CSV dataset.",
        show_default=True,
    ),
    target: str = typer.Option(
        "churn",
        "--target",
        help="Target column name.",
        show_default=True,
    ),
    provider: str = typer.Option(
        "gemini",
        "--provider",
        help="LLM provider: gemini | groq | ollama.",
        show_default=True,
    ),
):
    """Run the AutoML pipeline end-to-end on a CSV dataset."""
    from scripts.run_pipeline import run_pipeline

    console.print(Panel(
        f"[bold]Dataset:[/bold] {csv}\n"
        f"[bold]Target:[/bold]  {target}\n"
        f"[bold]Provider:[/bold] {provider}",
        title="[bold cyan]Nimbus AutoML - Starting Run[/bold cyan]",
        expand=False,
    ))

    try:
        final_state = run_pipeline(csv_path=csv, target=target, provider=provider)
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Run failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    console.print(Panel(
        f"[green bold]Best model:[/green bold]  {final_state.get('best_model_id')}\n"
        f"[green bold]Model bundle:[/green bold] {final_state.get('model_path') or 'N/A (export failed)'}\n"
        f"[green bold]Report:[/green bold]       {final_state.get('report_path')}",
        title="[bold green]Run Complete[/bold green]",
        expand=False,
    ))


# ---------------------------------------------------------------------------
# stress-test
# ---------------------------------------------------------------------------

@app.command("stress-test")
def stress_test():
    """Run the pipeline across all local datasets and print a summary table."""
    from scripts.stress_test import run_stress_test

    console.print(Panel(
        "Running pipeline against all datasets in [cyan]data/raw/[/cyan]...",
        title="[bold cyan]Nimbus - Stress Test[/bold cyan]",
        expand=False,
    ))

    try:
        summary = run_stress_test()
    except Exception as e:
        console.print(f"[bold red]Stress test failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    table = Table(title="Stress Test Results", show_lines=True)
    table.add_column("Dataset ID", style="cyan")
    table.add_column("Status")
    table.add_column("Best Model")
    table.add_column("Time (s)")

    for r in summary["results"]:
        status_str = "[green]PASS[/green]" if r["status"] == "Success" else "[red]FAIL[/red]"
        table.add_row(r["id"], status_str, r.get("best_model", "N/A"), f"{r.get('time_sec', 0):.1f}")

    console.print(table)
    console.print(f"\n[bold]Full report:[/bold] {summary['report_path']}")

    if not summary["all_passed"]:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# verify-providers
# ---------------------------------------------------------------------------

@app.command("verify-providers")
def verify_providers():
    """Check that each configured LLM provider is reachable."""
    from scripts.verify_providers import verify_providers as _verify

    console.print("[bold cyan]Checking LLM provider connectivity...[/bold cyan]")
    results = _verify()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("Outcome")

    any_failed = False
    for provider, outcome in results.items():
        if "FAILED" in outcome:
            any_failed = True
            table.add_row(provider, f"[red]{outcome}[/red]")
        elif "SKIPPED" in outcome:
            table.add_row(provider, f"[yellow]{outcome}[/yellow]")
        else:
            table.add_row(provider, f"[green]{outcome}[/green]")

    console.print(table)

    if any_failed:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# generate-data
# ---------------------------------------------------------------------------

@app.command("generate-data")
def generate_data(
    output_dir: str = typer.Option(
        "data/raw",
        "--output-dir",
        help="Directory to write the synthetic CSV.",
        show_default=True,
    ),
    n_rows: int = typer.Option(
        2000,
        "--n-rows",
        help="Number of rows to generate.",
        show_default=True,
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility.",
        show_default=True,
    ),
):
    """Generate the synthetic ground-truth dataset."""
    from scripts.generate_synthetic import generate_synthetic_dataset

    out = Path(output_dir)
    console.print(f"Generating synthetic dataset -> [cyan]{out}[/cyan] ({n_rows} rows, seed={seed})...")
    try:
        meta = generate_synthetic_dataset(out, root=ROOT, n_rows=n_rows, seed=seed)
        console.print(f"[green]SUCCESS:[/green] Wrote [bold]{meta['path']}[/bold] ({meta['n_rows']} rows, {meta['n_cols']} cols)")
        console.print(f"  Ground truth: {meta['ground_truth_path']}")
    except Exception as e:
        console.print(f"[bold red]Failed:[/bold red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# download-data
# ---------------------------------------------------------------------------

@app.command("download-data")
def download_data():
    """Download Titanic, Wine Quality, and California Housing CSVs."""
    from scripts.download_datasets import main as _download_main

    console.print("[bold cyan]Downloading public datasets...[/bold cyan]")
    try:
        _download_main()
        console.print("[green]All datasets ready.[/green]")
    except Exception as e:
        console.print(f"[bold red]Download failed:[/bold red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# serve-mcp
# ---------------------------------------------------------------------------

@app.command("serve-mcp")
def serve_mcp(
    transport: str = typer.Option(
        "streamable-http",
        "--transport",
        help="MCP transport: streamable-http (default) or stdio.",
        show_default=True,
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="HTTP port (only used for streamable-http transport).",
        show_default=True,
    ),
):
    """Start the MCP server exposing pipeline tools.

    [bold]Default[/bold]: streamable-http on port 8000 — reachable by remote clients.
    Use [cyan]--transport stdio[/cyan] for local Claude Desktop subprocess integration.
    """
    from automl_agents.mcp_server import mcp

    if transport == "streamable-http":
        console.print(Panel(
            f"Listening on [bold cyan]http://localhost:{port}[/bold cyan]\n"
            "Tools: run_automl_pipeline, get_run_report, list_local_datasets, run_stress_test",
            title="[bold cyan]Nimbus MCP Server - streamable-http[/bold cyan]",
            expand=False,
        ))
        mcp.settings.port = port
        mcp.run(transport="streamable-http")
    else:
        console.print("[dim]Starting Nimbus MCP server over stdio...[/dim]")
        mcp.run(transport="stdio")
