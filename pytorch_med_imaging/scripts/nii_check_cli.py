#!/usr/bin/env python
"""
A tool to calculate and cache hash values of NIFTI file pixel data,
and detect duplicate pixel data across files.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import click
import SimpleITK as sitk
import pandas as pd
import concurrent.futures
from tqdm import tqdm
try:
    from mpi4py import MPI
    from tqdm.contrib.concurrent import process_map
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
from tabulate import tabulate

CACHE_FILE = '.nifti_hash_cache.json'

def load_cache(cache_path: Path) -> Dict:
    """Load existing cache if available, otherwise return empty dict"""
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache_path: Path, cache_data: Dict) -> None:
    """Save cache data to json file"""
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

def hash_nifti_data(file_path: Path) -> str:
    """Calculate SHA256 hash of NIFTI pixel data"""
    img = sitk.ReadImage(str(file_path))
    pixel_array = sitk.GetArrayFromImage(img)
    return hashlib.sha256(pixel_array.tobytes()).hexdigest()

def analyze_duplicates(cache: Dict) -> pd.DataFrame:
    """
    Analyze cache data to find duplicates and create a summary DataFrame
    """
    # Group files by hash
    hash_groups = defaultdict(list)
    for file_path, data in cache.items():
        hash_groups[data['hash']].append(file_path)

    # Create records for DataFrame
    records = []
    for hash_val, files in hash_groups.items():
        if len(files) > 1:  # Only include groups with duplicates
            for file_path in files:
                records.append({
                    'file_path': file_path,
                    'hash': hash_val,
                    'duplicate_group_size': len(files),
                    'duplicate_files': ','.join(f for f in files if f != file_path)
                })

    # Create and sort DataFrame
    if records:
        df = pd.DataFrame(records)
        df = df.sort_values(['duplicate_group_size', 'hash'], ascending=[False, True])
        return df
    else:
        return pd.DataFrame(columns=['file_path', 'hash', 'duplicate_group_size', 'duplicate_files'])

@click.group()
@click.pass_context
def cli(ctx):
    """Batch management CLI."""
    ctx.ensure_object(dict)

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--force', is_flag=True, help='Force rehash all files ignoring cache')
@click.option('--output', '-o', type=click.Path(), help='Save duplicate analysis to CSV file')
@click.option('--debug', is_flag=True, help='Debug mode: only process first 10 files')
@click.option('--workers', default=None, type=int, help='Number of worker processes (default: cpu count)')
@click.option('--mpi', is_flag=True, help='Enable MPI multiprocessing mode')
@click.pass_context
def hasher(ctx, directory: str, force: bool, output: str, debug: bool, workers: int, mpi: bool) -> None:
    """Process all NIFTI files in directory and analyze duplicate pixel data"""
    dir_path = Path(directory)
    cache_path = dir_path / CACHE_FILE

    # Load existing cache
    cache = load_cache(cache_path) if not force else {}

    # Find all .nii and .nii.gz files
    nifti_files = []
    for ext in ['.nii', '.nii.gz']:
        nifti_files.extend(dir_path.glob(f'**/*{ext}'))
    nifti_files = sorted(nifti_files)

    # Debug mode: only process first 10 files
    if debug:
        nifti_files = nifti_files[:10]
        click.echo(f'DEBUG MODE: Only processing first {len(nifti_files)} files.')

    # Prepare list of files to process (skip unchanged if not force)
    files_to_process = []
    for file_path in nifti_files:
        rel_path = str(file_path.relative_to(dir_path))
        mtime = os.path.getmtime(file_path)
        if not force and rel_path in cache and cache[rel_path]['mtime'] == mtime:
            continue
        files_to_process.append((file_path, rel_path, mtime))

    def process_file(args):
        file_path, rel_path, mtime = args
        try:
            file_hash = hash_nifti_data(file_path)
            return (rel_path, file_hash, mtime, None)
        except Exception as e:
            return (rel_path, None, mtime, str(e))

    results = []
    if files_to_process:
        click.echo(f"Processing {len(files_to_process)} files...")

        if mpi:
            if not MPI_AVAILABLE:
                click.echo("Error: --mpi specified but mpi4py/tqdm.contrib.concurrent not available.", err=True)
                return
            results = process_map(process_file, files_to_process, max_workers=workers, desc="Hashing", chunksize=1)
        elif workers is None or workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for res in tqdm(executor.map(process_file, files_to_process), total=len(files_to_process), desc="Hashing"):
                    results.append(res)
        else:
            # Single-threaded mode
            for args in tqdm(files_to_process, desc="Hashing"):
                results.append(process_file(args))

        # Update cache with results
        for rel_path, file_hash, mtime, error in results:
            if error:
                click.echo(f'Error processing {rel_path}: {error}', err=True)
            else:
                cache[rel_path] = {
                    'hash': file_hash,
                    'mtime': mtime
                }
                click.echo(f'Processed: {rel_path}')
    else:
        click.echo('No new or changed files to process.')

    # Save updated cache
    save_cache(cache_path, cache)
    click.echo(f'Cache saved to {cache_path}')

    # Analyze duplicates
    df_duplicates = analyze_duplicates(cache)

    if len(df_duplicates) > 0:
        click.echo("\nDuplicate files found:")
        click.echo(f"Total duplicate groups: {df_duplicates['hash'].nunique()}")
        click.echo(f"Total files with duplicates: {len(df_duplicates)}")

        # Display summary
        click.echo("\nDuplicate groups summary:")
        # print(df_duplicates.to_string(index=False))
        print(tabulate(df_duplicates, headers='keys', showindex=False, tablefmt='simple'))

        # Save to CSV if output path provided
        if output:
            df_duplicates.to_csv(output, index=False)
            click.echo(f"\nDetailed analysis saved to: {output}")
    else:
        click.echo("\nNo duplicate files found.")

if __name__ == '__main__':
    cli()