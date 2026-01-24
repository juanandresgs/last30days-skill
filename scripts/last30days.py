#!/usr/bin/env python3
"""
last30days - Research a topic from the last 30 days on Reddit + X.

Usage:
    python3 last30days.py <topic> [options]

Options:
    --mock              Use fixtures instead of real API calls
    --emit=MODE         Output mode: compact|json|md|context|path (default: compact)
    --sources=MODE      Source selection: auto|reddit|x|both (default: auto)
    --quick             Faster research with fewer sources (8-12 each)
    --deep              Comprehensive research with more sources (50-70 Reddit, 40-60 X)
    --debug             Enable verbose debug logging
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add lib to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from lib import (
    dates,
    dedupe,
    env,
    http,
    models,
    normalize,
    openai_reddit,
    reddit_enrich,
    render,
    schema,
    score,
    ui,
    websearch,
    xai_x,
)


def load_fixture(name: str) -> dict:
    """Load a fixture file."""
    fixture_path = SCRIPT_DIR.parent / "fixtures" / name
    if fixture_path.exists():
        with open(fixture_path) as f:
            return json.load(f)
    return {}


def run_research(
    topic: str,
    sources: str,
    config: dict,
    selected_models: dict,
    from_date: str,
    to_date: str,
    depth: str = "default",
    mock: bool = False,
    progress: ui.ProgressDisplay = None,
) -> tuple:
    """Run the research pipeline.

    Returns:
        Tuple of (reddit_items, x_items, web_needed, raw_openai, raw_xai, raw_reddit_enriched, reddit_error, x_error)

    Note: web_needed is True when WebSearch should be performed by Claude.
    The script outputs a marker and Claude handles WebSearch in its session.
    """
    reddit_items = []
    x_items = []
    raw_openai = None
    raw_xai = None
    raw_reddit_enriched = []
    reddit_error = None
    x_error = None

    # Check if WebSearch is needed
    web_needed = sources in ("all", "web", "reddit-web", "x-web")

    # Reddit search via OpenAI
    if sources in ("both", "reddit", "all", "reddit-web"):
        if progress:
            progress.start_reddit()

        if mock:
            raw_openai = load_fixture("openai_sample.json")
        else:
            try:
                raw_openai = openai_reddit.search_reddit(
                    config["OPENAI_API_KEY"],
                    selected_models["openai"],
                    topic,
                    depth=depth,
                )
            except http.HTTPError as e:
                if progress:
                    progress.show_error(f"Reddit API failed: {e}")
                raw_openai = {"error": str(e)}
                reddit_error = f"API error: {e}"
            except Exception as e:
                if progress:
                    progress.show_error(f"Reddit error: {e}")
                raw_openai = {"error": str(e)}
                reddit_error = f"{type(e).__name__}: {e}"

        # Parse response
        reddit_items = openai_reddit.parse_reddit_response(raw_openai)

        if progress:
            progress.end_reddit(len(reddit_items))

        # Enrich with real Reddit data
        if reddit_items:
            if progress:
                progress.start_reddit_enrich(1, len(reddit_items))

            for i, item in enumerate(reddit_items):
                if progress and i > 0:
                    progress.update_reddit_enrich(i + 1, len(reddit_items))

                if mock:
                    mock_thread = load_fixture("reddit_thread_sample.json")
                    reddit_items[i] = reddit_enrich.enrich_reddit_item(item, mock_thread)
                else:
                    reddit_items[i] = reddit_enrich.enrich_reddit_item(item)

                raw_reddit_enriched.append(reddit_items[i])

            if progress:
                progress.end_reddit_enrich()

    # X search via xAI
    if sources in ("both", "x", "all", "x-web"):
        if progress:
            progress.start_x()

        if mock:
            raw_xai = load_fixture("xai_sample.json")
        else:
            try:
                raw_xai = xai_x.search_x(
                    config["XAI_API_KEY"],
                    selected_models["xai"],
                    topic,
                    from_date,
                    to_date,
                    depth=depth,
                )
            except http.HTTPError as e:
                if progress:
                    progress.show_error(f"X API failed: {e}")
                raw_xai = {"error": str(e)}
                x_error = f"API error: {e}"
            except Exception as e:
                if progress:
                    progress.show_error(f"X error: {e}")
                raw_xai = {"error": str(e)}
                x_error = f"{type(e).__name__}: {e}"

        # Parse response
        x_items = xai_x.parse_x_response(raw_xai)

        if progress:
            progress.end_x(len(x_items))

    return reddit_items, x_items, web_needed, raw_openai, raw_xai, raw_reddit_enriched, reddit_error, x_error


def main():
    parser = argparse.ArgumentParser(
        description="Research a topic from the last 30 days on Reddit + X"
    )
    parser.add_argument("topic", nargs="?", help="Topic to research")
    parser.add_argument("--mock", action="store_true", help="Use fixtures")
    parser.add_argument(
        "--emit",
        choices=["compact", "json", "md", "context", "path"],
        default="compact",
        help="Output mode",
    )
    parser.add_argument(
        "--sources",
        choices=["auto", "reddit", "x", "both"],
        default="auto",
        help="Source selection",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Faster research with fewer sources (8-12 each)",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Comprehensive research with more sources (50-70 Reddit, 40-60 X)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--include-web",
        action="store_true",
        help="Include general web search alongside Reddit/X (lower weighted)",
    )

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        os.environ["LAST30DAYS_DEBUG"] = "1"
        # Re-import http to pick up debug flag
        from lib import http as http_module
        http_module.DEBUG = True

    # Determine depth
    if args.quick and args.deep:
        print("Error: Cannot use both --quick and --deep", file=sys.stderr)
        sys.exit(1)
    elif args.quick:
        depth = "quick"
    elif args.deep:
        depth = "deep"
    else:
        depth = "default"

    if not args.topic:
        print("Error: Please provide a topic to research.", file=sys.stderr)
        print("Usage: python3 last30days.py <topic> [options]", file=sys.stderr)
        sys.exit(1)

    # Load config
    config = env.get_config()

    # Check available sources
    available = env.get_available_sources(config)

    # Mock mode can work without keys
    if args.mock:
        if args.sources == "auto":
            sources = "both"
        else:
            sources = args.sources
    else:
        # Validate requested sources against available
        sources, error = env.validate_sources(args.sources, available, args.include_web)
        if error:
            # If it's a warning about WebSearch fallback, print but continue
            if "WebSearch fallback" in error:
                print(f"Note: {error}", file=sys.stderr)
            else:
                print(f"Error: {error}", file=sys.stderr)
                sys.exit(1)

    # Get date range
    from_date, to_date = dates.get_date_range(30)

    # Initialize progress display
    progress = ui.ProgressDisplay(args.topic, show_banner=True)

    # Select models
    if args.mock:
        # Use mock models
        mock_openai_models = load_fixture("models_openai_sample.json").get("data", [])
        mock_xai_models = load_fixture("models_xai_sample.json").get("data", [])
        selected_models = models.get_models(
            {
                "OPENAI_API_KEY": "mock",
                "XAI_API_KEY": "mock",
                **config,
            },
            mock_openai_models,
            mock_xai_models,
        )
    else:
        selected_models = models.get_models(config)

    # Determine mode string
    if sources == "all":
        mode = "all"  # reddit + x + web
    elif sources == "both":
        mode = "both"  # reddit + x
    elif sources == "reddit":
        mode = "reddit-only"
    elif sources == "reddit-web":
        mode = "reddit-web"
    elif sources == "x":
        mode = "x-only"
    elif sources == "x-web":
        mode = "x-web"
    elif sources == "web":
        mode = "web-only"
    else:
        mode = sources

    # Run research
    reddit_items, x_items, web_needed, raw_openai, raw_xai, raw_reddit_enriched, reddit_error, x_error = run_research(
        args.topic,
        sources,
        config,
        selected_models,
        from_date,
        to_date,
        depth,
        args.mock,
        progress,
    )

    # Processing phase
    progress.start_processing()

    # Normalize items
    normalized_reddit = normalize.normalize_reddit_items(reddit_items, from_date, to_date)
    normalized_x = normalize.normalize_x_items(x_items, from_date, to_date)

    # Score items
    scored_reddit = score.score_reddit_items(normalized_reddit)
    scored_x = score.score_x_items(normalized_x)

    # Sort items
    sorted_reddit = score.sort_items(scored_reddit)
    sorted_x = score.sort_items(scored_x)

    # Dedupe items
    deduped_reddit = dedupe.dedupe_reddit(sorted_reddit)
    deduped_x = dedupe.dedupe_x(sorted_x)

    progress.end_processing()

    # Create report
    report = schema.create_report(
        args.topic,
        from_date,
        to_date,
        mode,
        selected_models.get("openai"),
        selected_models.get("xai"),
    )
    report.reddit = deduped_reddit
    report.x = deduped_x
    report.reddit_error = reddit_error
    report.x_error = x_error

    # Generate context snippet
    report.context_snippet_md = render.render_context_snippet(report)

    # Write outputs
    render.write_outputs(report, raw_openai, raw_xai, raw_reddit_enriched)

    # Show completion
    progress.show_complete(len(deduped_reddit), len(deduped_x))

    # Output result
    output_result(report, args.emit, web_needed, args.topic, from_date, to_date)


def output_result(
    report: schema.Report,
    emit_mode: str,
    web_needed: bool = False,
    topic: str = "",
    from_date: str = "",
    to_date: str = "",
):
    """Output the result based on emit mode."""
    if emit_mode == "compact":
        print(render.render_compact(report))
    elif emit_mode == "json":
        print(json.dumps(report.to_dict(), indent=2))
    elif emit_mode == "md":
        print(render.render_full_report(report))
    elif emit_mode == "context":
        print(report.context_snippet_md)
    elif emit_mode == "path":
        print(render.get_context_path())

    # Output WebSearch instructions if needed
    if web_needed:
        print("\n" + "="*60)
        print("### WEBSEARCH REQUIRED ###")
        print("="*60)
        print(f"Topic: {topic}")
        print(f"Date range: {from_date} to {to_date}")
        print("")
        print("Claude: Use your WebSearch tool to find 8-15 relevant web pages.")
        print("EXCLUDE: reddit.com, x.com, twitter.com (already covered above)")
        print("INCLUDE: blogs, docs, news, tutorials from the last 30 days")
        print("")
        print("After searching, synthesize WebSearch results WITH the Reddit/X")
        print("results above. WebSearch items should rank LOWER than comparable")
        print("Reddit/X items (they lack engagement metrics).")
        print("="*60)


if __name__ == "__main__":
    main()
