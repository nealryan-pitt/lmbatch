#!/usr/bin/env python3
"""LM Batch: Batch process text files through LM Studio's local LLM server."""

import sys
import os
from pathlib import Path
import click
from typing import List

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from processor import BatchProcessor
from file_manager import FileManager


@click.command()
@click.option('--prompt', '-p', 
              required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to prompt file')
@click.option('--input', '-i', 
              required=True,
              type=click.Path(exists=True),
              help='Path to text file(s) or directory')
@click.option('--output', '-o', 
              default='output',
              type=click.Path(),
              help='Output directory (default: output/)')
@click.option('--server', 
              default='http://localhost:1234',
              help='LM Studio server URL (default: http://localhost:1234)')
@click.option('--model',
              default='gpt-oss-20b',
              help='Model name to use (default: gpt-oss-20b)')
@click.option('--temperature',
              type=float,
              default=0.1,
              help='Sampling temperature (0.0-1.0, default: 0.1)')
@click.option('--max-tokens',
              type=int,
              default=32000,
              help='Maximum response tokens (default: 32000)')
@click.option('--concurrent',
              type=int,
              default=3,
              help='Number of concurrent requests (default: 3)')
@click.option('--max-context',
              type=int,
              default=16384,
              help='Maximum context length in tokens (default: 16384, 0=auto-detect)')
@click.option('--strategy',
              type=click.Choice(['fail', 'truncate', 'split', 'force']),
              default='force',
              help='How to handle oversized content (default: force)')
@click.option('--auto-detect-context',
              is_flag=True,
              help='Auto-detect model context length')
@click.option('--overlap-tokens',
              type=int,
              default=300,
              help='Overlap tokens for chunking (default: 300)')
@click.option('--safety-margin',
              type=int,
              default=500,
              help='Safety margin tokens (default: 500)')
@click.option('--ctx-size',
              type=int,
              default=16384,
              help='Context size for LM Studio model (default: 16384)')
@click.option('--config',
              type=click.Path(),
              help='Configuration file path (default: config.yaml)')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Verbose output')
@click.option('--dry-run',
              is_flag=True,
              help='Show what would be processed without making API calls')
@click.option('--overwrite',
              is_flag=True,
              help='Overwrite existing output files')
def main(prompt, input, output, server, model, temperature, max_tokens, 
         concurrent, max_context, strategy, auto_detect_context, overlap_tokens, 
         safety_margin, ctx_size, config, verbose, dry_run, overwrite):
    """Batch process text files through LM Studio's local LLM server.
    
    This tool takes a prompt file and processes one or more text files,
    sending each combined prompt+text to LM Studio and saving the responses.
    
    Example usage:
    
        python main.py --prompt promptfiles/analyze.txt --input txtfiles/
        
        python main.py -p prompts/summarize.txt -i document.txt -o results/
    """
    
    # Initialize configuration
    try:
        cfg = Config(config_path=config)
        
        # Override config with command line arguments
        cfg.set('lm_studio', 'server_url', server)
        cfg.set('lm_studio', 'model', model)
        cfg.set('processing', 'temperature', temperature)
        cfg.set('processing', 'max_tokens', max_tokens)
        cfg.set('processing', 'concurrent_requests', concurrent)
        cfg.set('processing', 'max_context_length', max_context)
        cfg.set('context_handling', 'strategy', strategy)
        cfg.set('context_handling', 'auto_detect', auto_detect_context)
        cfg.set('context_handling', 'overlap_tokens', overlap_tokens)
        cfg.set('context_handling', 'safety_margin', safety_margin)
        cfg.set('context_handling', 'ctx_size', ctx_size)
        cfg.set('output', 'directory', output)
        cfg.set('output', 'overwrite', overwrite)
        
        # Sync max_context with ctx_size if ctx_size was explicitly set
        # This ensures the application's context limit matches the model's context size
        if ctx_size != 16384:  # If ctx_size was changed from default
            max_context = ctx_size
            cfg.set('processing', 'max_context_length', max_context)
        
        # Handle special case for max_context = 0 (auto-detect)
        if max_context == 0:
            cfg.set('context_handling', 'auto_detect', True)
        else:
            cfg._override_context_length = max_context
        
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)
    
    # Initialize processor
    try:
        processor = BatchProcessor(cfg, verbose=verbose)
    except Exception as e:
        click.echo(f"Error initializing processor: {e}", err=True)
        sys.exit(1)
    
    # Validate setup
    if verbose:
        click.echo("Validating setup...")
    
    validation = processor.validate_setup()
    if not validation['valid']:
        click.echo("Setup validation failed:", err=True)
        for error in validation['errors']:
            click.echo(f"  • {error}", err=True)
        sys.exit(1)
    
    if verbose:
        client_info = validation['client_status']
        click.echo(f"✓ Connected to LM Studio at {client_info['server_url']}")
        if client_info['models']:
            click.echo(f"  Available models: {', '.join(client_info['models'])}")
        click.echo(f"✓ Output directory ready: {output}")
    
    # Find input files
    try:
        file_manager = FileManager()
        input_files = file_manager.find_text_files(input)
        
        if verbose:
            click.echo(f"Found {len(input_files)} text file(s) to process")
            if len(input_files) <= 10:
                for file_path in input_files:
                    click.echo(f"  • {file_path}")
            else:
                for file_path in input_files[:5]:
                    click.echo(f"  • {file_path}")
                click.echo(f"  ... and {len(input_files) - 5} more files")
    
    except Exception as e:
        click.echo(f"Error finding input files: {e}", err=True)
        sys.exit(1)
    
    # Process files
    def progress_callback(current, total, result):
        if verbose and result:
            status = "✓" if result['success'] else "✗"
            filename = Path(result['file_path']).name
            click.echo(f"  {status} {filename}")
    
    try:
        if dry_run:
            click.echo("\n=== DRY RUN MODE ===")
            click.echo(f"Would process {len(input_files)} files with prompt: {prompt}")
            click.echo(f"Output would go to: {output}")
            click.echo(f"Server: {server}")
            click.echo(f"Model: {model}")
            click.echo(f"Temperature: {temperature}")
            click.echo(f"Max tokens: {max_tokens}")
            click.echo(f"Context size: {ctx_size}")
            
            # Show context handling info
            if auto_detect_context or max_context == 0:
                detected_context = cfg.get_model_context_length(model)
                click.echo(f"Context length: {detected_context:,} tokens (auto-detected)")
            else:
                click.echo(f"Context length: {max_context:,} tokens")
            
            click.echo(f"Strategy: {strategy}")
            if strategy == 'split':
                click.echo(f"Overlap tokens: {overlap_tokens}")
            click.echo(f"Safety margin: {safety_margin} tokens")
            click.echo("No files will be processed in dry run mode.")
        else:
            context_info = ""
            if auto_detect_context or max_context == 0:
                detected_context = cfg.get_model_context_length(model)
                context_info = f" (context: {detected_context:,}, strategy: {strategy})"
            else:
                context_info = f" (context: {max_context:,}, strategy: {strategy})"
            
            click.echo(f"\nProcessing {len(input_files)} files...{context_info}")
        
        results = processor.process_files(
            prompt_path=prompt,
            input_paths=input_files,
            progress_callback=progress_callback if verbose else None,
            dry_run=dry_run
        )
        
        if not results['success']:
            click.echo(f"Processing failed: {results['error']}", err=True)
            if 'details' in results:
                for detail in results['details']:
                    click.echo(f"  • {detail}", err=True)
            sys.exit(1)
        
        # Show results
        if dry_run:
            click.echo("\n✓ Dry run completed successfully")
        else:
            summary = processor.get_processing_summary()
            click.echo(f"\n=== Processing Complete ===")
            click.echo(f"Files processed: {summary['processed_files']}/{summary['total_files']}")
            click.echo(f"Success rate: {summary['success_rate']:.1f}%")
            
            if summary['failed_files'] > 0:
                click.echo(f"Failed files: {summary['failed_files']}")
                
            click.echo(f"Total tokens used: {summary['total_tokens']:,}")
            click.echo(f"Processing time: {summary['processing_time']:.1f}s")
            click.echo(f"Average per file: {summary['average_time_per_file']:.1f}s")
            
            output_summary = results.get('output_summary', {})
            if output_summary.get('file_count', 0) > 0:
                click.echo(f"\nOutput files created in: {output_summary['output_dir']}")
                click.echo(f"Total output files: {output_summary['file_count']}")
                
                # Show chunking summary if applicable
                total_chunks = sum(result.get('chunks_processed', 1) for result in results.get('results', []) if result['success'])
                if total_chunks > summary['processed_files']:
                    click.echo(f"Files were chunked: {total_chunks} total chunks processed")
                
                if verbose and len(output_summary.get('files', [])) <= 10:
                    for filename in output_summary['files']:
                        click.echo(f"  • {filename}")
            
            # Show errors if any
            if summary['errors'] and verbose:
                click.echo("\nErrors encountered:")
                for error in summary['errors'][:5]:  # Show first 5 errors
                    click.echo(f"  • {error}")
                if len(summary['errors']) > 5:
                    click.echo(f"  ... and {len(summary['errors']) - 5} more errors")
    
    except KeyboardInterrupt:
        click.echo("\n\nProcessing interrupted by user", err=True)
        sys.exit(1)
    
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)
    
    finally:
        # Cleanup
        processor.cleanup()


if __name__ == "__main__":
    main()
