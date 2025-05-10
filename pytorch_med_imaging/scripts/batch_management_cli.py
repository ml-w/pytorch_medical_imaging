#! /usr/bin/env python3

import pandas as pd
import click
from configparser import ConfigParser
from tabulate import tabulate
from pathlib import Path
from pprint import pprint, pformat

import rich
from rich.traceback import install

install(show_locals=True)


def get_batch_data(path: Path) -> dict:
    """Extract batch data from the given path.

    Args:
        path: Path to the batch directory

    Returns:
        dict: Dictionary containing batch data with the following structure:
            {
                'batch_name': {
                    'training': list,
                    'testing': list
                },
                'Validation': list
            }
    """
    batches = {}

    # Read regular batch files
    ini_files = list(path.glob('*.ini'))
    if not ini_files:
        raise click.ClickException(f"No ini files found in {path}")

    for ini_file in ini_files:
        config = ConfigParser()
        config.read(ini_file)
        training = config['FileList']['training']
        testing = config['FileList']['testing']
        batches[ini_file.stem] = {
            'training': training.split(',') if training else [],
            'testing': testing.split(',') if testing else []
        }

        # Read validation file if exists
    val_file = path / 'Validation.txt'
    if val_file.exists():
        with open(val_file, 'r') as f:
            val_batches = [r.rstrip() for r in f.readlines()]
        batches['Validation'] = val_batches

    return batches


@click.group()
@click.pass_context
def cli(ctx):
    """Batch management CLI."""
    ctx.ensure_object(dict)


@cli.command()
@click.option('--path', required=True, type=click.Path(exists=True, path_type=Path), help='Path to the batch file.')
@click.option('--details', is_flag=True,
              help='Display details of each batch. Othewise, display only the number of cases in each batch.')
@click.option('--max-cols', type=int, default=60, help='Maximum number of columns to display.')
@click.option('--table-style', type=str, default='heavy_grid', help='Table style to use.')
@click.pass_context
def display(ctx, path, details, max_cols, table_style):
    """Display all batches."""
    click.echo(f"Displaying all batches from {path}...")

    # Get batch data
    batches = get_batch_data(path)

    # Create a formatted version for display
    table_data = []
    batch_totals = {}

    # Add regular batches
    for batch_name, batch_data in batches.items():
        if batch_name != 'Validation':
            training_count = len(batch_data['training'])
            testing_count = len(batch_data['testing'])
            total_count = training_count + testing_count
            batch_totals[batch_name] = total_count

            if details:
                table_data.append([
                    batch_name,
                    ', '.join(batch_data['training']),
                    ', '.join(batch_data['testing']),
                    total_count
                ])
            else:
                table_data.append([
                    batch_name,
                    training_count,
                    testing_count,
                    total_count
                ])

    # Add Validation as a separate batch
    if 'Validation' in batches:
        validation_count = len(batches['Validation'])
        if details:
            table_data.append([
                'Validation',
                ', '.join(batches['Validation']),
                '',
                validation_count
            ])
        else:
            table_data.append([
                'Validation',
                validation_count,
                '',
                validation_count
            ])

    rich.print(
        tabulate(
            table_data,
            headers=['Batch', 'Training', 'Testing', 'Total'],
            tablefmt=table_style,
            maxcolwidths=[30, max_cols, max_cols, 10]
        )
    )


@cli.command()
@click.option('--path', required=True, type=click.Path(exists=True, path_type=Path), help='Path to the batch file.')
@click.argument('batch_name', type=str)
@click.argument('sub_category', type=click.Choice(['training', 'testing', 'validation']))
@click.argument('files', nargs=-1, type=click.Path(exists=True))
@click.pass_context
def add(ctx, path, batch_name, sub_category, files):
    """Add or update files in a batch.

    Arguments:
        batch_name: Name of the batch to add/update
        sub_category: Category to add files to (training/testing/validation)
        files: List of files to add
    """
    if not files:
        raise click.ClickException("No files provided")

    # Convert files to relative paths
    files = [str(Path(f).relative_to(path)) for f in files]

    if batch_name == 'all':
        # Get all batch names from ini files
        batch_names = [f.stem for f in path.glob('*.ini')]
        if not batch_names:
            raise click.ClickException("No batches found in the directory")

        click.echo(f"Adding files to all batches ({len(batch_names)} batches) {sub_category}...")

        # Process each batch
        for name in batch_names:
            ini_file = path / f"{name}.ini"
            config = ConfigParser()
            config.read(ini_file)

            # Get existing files
            existing_files = set(config['FileList'][sub_category].split(',')) if config['FileList'][
                sub_category] else set()

            # Add new files
            existing_files.update(files)

            # Update config
            config['FileList'][sub_category] = ','.join(sorted(existing_files))

            # Write back to file
            with open(ini_file, 'w') as f:
                config.write(f)

        click.echo(f"Updated all batches {sub_category} with {len(files)} files")
    else:
        click.echo(f"Adding files to {batch_name} {sub_category}...")

        if sub_category == 'validation':
            # Handle validation files
            val_file = path / 'Validation.txt'
            existing_files = set()
            if val_file.exists():
                with open(val_file, 'r') as f:
                    existing_files = set(line.strip() for line in f)

            # Add new files
            existing_files.update(files)

            # Write back to file
            with open(val_file, 'w') as f:
                for file in sorted(existing_files):
                    f.write(f"{file}\n")

            click.echo(f"Updated validation set with {len(files)} files")
        else:
            # Handle training/testing files
            ini_file = path / f"{batch_name}.ini"
            config = ConfigParser()

            if ini_file.exists():
                config.read(ini_file)
            else:
                config['FileList'] = {'training': '', 'testing': ''}

            # Get existing files
            existing_files = set(config['FileList'][sub_category].split(',')) if config['FileList'][
                sub_category] else set()

            # Add new files
            existing_files.update(files)

            # Update config
            config['FileList'][sub_category] = ','.join(sorted(existing_files))

            # Write back to file
            with open(ini_file, 'w') as f:
                config.write(f)

            click.echo(f"Updated {batch_name} {sub_category} with {len(files)} files")


@cli.command()
@click.option('--path', required=True, type=click.Path(exists=True, path_type=Path), help='Path to the batch file.')
@click.argument('batch_name', type=str)
@click.argument('case_ids', nargs=-1, type=str)
@click.pass_context
def remove(ctx, path, batch_name, case_ids):
    """Remove case IDs from a batch.

    Args:
        path: Path to the batch directory
        batch_name: Name of the batch to remove case IDs from, or 'all' to remove from all batches
        case_ids: List of case IDs to remove, or a single CSV string of case IDs
    """
    if not case_ids:
        raise click.ClickException("No case IDs provided")

    # If only one argument is provided and it contains commas, treat it as a CSV string
    if len(case_ids) == 1 and ',' in case_ids[0]:
        case_ids = [cid.strip() for cid in case_ids[0].split(',')]

    if batch_name == 'all':
        # Get all batch names from ini files
        batch_names = [f.stem for f in path.glob('*.ini')]
        if not batch_names:
            raise click.ClickException("No batches found in the directory")

        click.echo(f"Removing case IDs from all batches ({len(batch_names)} batches)...")

        # Process each batch
        for name in batch_names:
            ini_file = path / f"{name}.ini"
            config = ConfigParser()
            config.read(ini_file)

            # Remove case IDs from both training and testing
            for category in ['training', 'testing']:
                existing_cases = set(config['FileList'][category].split(',')) if config['FileList'][category] else set()
                existing_cases.difference_update(case_ids)
                config['FileList'][category] = ','.join(sorted(existing_cases))

            # Write back to file
            with open(ini_file, 'w') as f:
                config.write(f)

        # Also check and remove from validation if it exists
        val_file = path / 'Validation.txt'
        if val_file.exists():
            existing_cases = set()
            with open(val_file, 'r') as f:
                existing_cases = set(line.strip() for line in f)
            existing_cases.difference_update(case_ids)
            with open(val_file, 'w') as f:
                for case_id in sorted(existing_cases):
                    f.write(f"{case_id}\n")

        click.echo(f"Removed case IDs from all batches and validation set")
    else:
        click.echo(f"Removing case IDs from {batch_name}...")

        # Handle validation files
        val_file = path / 'Validation.txt'
        if val_file.exists():
            existing_cases = set()
            with open(val_file, 'r') as f:
                existing_cases = set(line.strip() for line in f)
            existing_cases.difference_update(case_ids)
            with open(val_file, 'w') as f:
                for case_id in sorted(existing_cases):
                    f.write(f"{case_id}\n")

        # Handle training/testing files
        ini_file = path / f"{batch_name}.ini"
        if not ini_file.exists():
            raise click.ClickException(f"Batch {batch_name} does not exist")

        config = ConfigParser()
        config.read(ini_file)

        # Remove case IDs from both training and testing
        for category in ['training', 'testing']:
            existing_cases = set(config['FileList'][category].split(',')) if config['FileList'][category] else set()
            existing_cases.difference_update(case_ids)
            config['FileList'][category] = ','.join(sorted(existing_cases))

        # Write back to file
        with open(ini_file, 'w') as f:
            config.write(f)

        click.echo(f"Removed case IDs from {batch_name} and validation set")


@cli.command()
@click.option('--path', required=True, type=click.Path(exists=True, path_type=Path), help='Path to the batch file.')
@click.argument('case_id', type=str)
@click.pass_context
def query(ctx, path, case_id):
    """Query a case ID and indicate which batch it belongs to

    Arguments:
        case_id: Case ID to query
    """
    batches = get_batch_data(path)
    out_msg = "Case {} found in batch {}"
    for batch_name, batch_data in batches.items():
        if batch_name == 'Validation':
            if case_id in batch_data:
                print(out_msg.format(case_id, batch_name))
                return 0
        else:
            if case_id in batch_data['training']:
                print(out_msg.format(case_id, batch_name) + f" under subgroup: training")
                return 0
            elif case_id in batch_data['testing']:
                print(out_msg.format(case_id, batch_name) + f" under subgroup: testing")
                return 0
    print(f"Case {case_id} not found in any batch")
    return 1


@cli.command()
@click.option('--path', required=True, type=click.Path(exists=True, path_type=Path), help='Path to the batch file.')
@click.option('--output', type=click.Path(path_type=Path), help='Output file path. If not provided, prints to stdout.')
@click.pass_context
def case2str(ctx, path, output):
    """Convert all case IDs to a sorted CSV string.

    Gets all unique case IDs from all batches and combines them into a single sorted list.
    Ignores batch divisions and returns all cases as a single CSV string.

    Arguments:
        path: Path to the batch directory
        output: Optional path to save the CSV. If not provided, prints to stdout.
    """
    batches = get_batch_data(path)
    all_cases = set()

    # Collect all cases from all batches
    for batch_name, batch_data in batches.items():
        if batch_name == 'Validation':
            all_cases.update(batch_data)
        else:
            all_cases.update(batch_data['training'])
            all_cases.update(batch_data['testing'])

    # Sort cases
    sorted_cases = sorted(all_cases)

    # Create CSV string
    csv_string = ','.join(sorted_cases)

    # Output
    if output:
        with open(output, 'w') as f:
            f.write(csv_string)
        click.echo(f"CSV string written to {output}")
    else:
        click.echo(csv_string)


if __name__ == '__main__':
    cli(obj={})