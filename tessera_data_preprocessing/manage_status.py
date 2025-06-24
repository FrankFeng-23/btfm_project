# 文件名: manage_status.py
import argparse
import csv
import os
import sys
from pathlib import Path

# 定义CSV文件的列名
FIELDNAMES = ['tiff_name', 'year', 'status']
# 定义各种状态
STATUS_UNPROCESSED = 'unprocessed'
STATUS_PROCESSING = 'processing'
STATUS_PROCESSED = 'processed'
STATUS_BROKEN = 'broken'

def init_csv(csv_path, source_dir, years):
    """
    初始化CSV文件。
    扫描TIFF源目录，为每个TIFF文件和年份的组合创建一条记录。
    如果CSV文件已存在，则不执行任何操作。
    """
    csv_file = Path(csv_path)
    if csv_file.exists():
        print(f"Status file '{csv_path}' already exists. Skipping initialization.", file=sys.stderr)
        return

    print(f"Creating and populating status file at '{csv_path}'...", file=sys.stderr)
    
    tiff_files = sorted([p.name.replace('.tiff', '') for p in Path(source_dir).glob('*.tiff')])
    if not tiff_files:
        print(f"Error: No TIFF files found in '{source_dir}'.", file=sys.stderr)
        sys.exit(1)

    with csv_file.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for tiff_name in tiff_files:
            for year in years:
                writer.writerow({
                    'tiff_name': tiff_name,
                    'year': year,
                    'status': STATUS_UNPROCESSED
                })
    print(f"Successfully initialized with {len(tiff_files) * len(years)} tasks.", file=sys.stderr)


def claim_task(csv_path):
    """
    认领一个任务。
    它会按以下顺序查找任务：
    1. 状态为 'unprocessed' 的任务
    2. 状态为 'broken' 的任务
    找到后，将状态更新为 'processing' 并返回 "tiff_name,year"。
    如果没有可用的任务，则返回 "NO_WORK_AVAILABLE"。
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Status file '{csv_path}' not found. Please run with 'init' command first.", file=sys.stderr)
        sys.exit(1)

    rows = []
    with csv_file.open('r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    task_to_claim = None
    
    # 优先处理未处理的
    for row in rows:
        if row['status'] == STATUS_UNPROCESSED:
            task_to_claim = row
            break
    
    # 如果没有未处理的，则处理失败的
    if not task_to_claim:
        for row in rows:
            if row['status'] == STATUS_BROKEN:
                task_to_claim = row
                break

    if task_to_claim:
        # 更新状态为 "processing"
        task_to_claim['status'] = STATUS_PROCESSING
        with csv_file.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        # 返回认领的任务信息
        print(f"{task_to_claim['tiff_name']},{task_to_claim['year']}")
    else:
        # 没有可处理的任务
        print("NO_WORK_AVAILABLE")


def update_status(csv_path, tiff_name, year, new_status):
    """
    更新指定任务的状态。
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Status file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    rows = []
    with csv_file.open('r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    task_found = False
    for row in rows:
        if row['tiff_name'] == tiff_name and row['year'] == year:
            row['status'] = new_status
            task_found = True
            break
            
    if not task_found:
        print(f"Warning: Task '{tiff_name}' for year '{year}' not found in status file.", file=sys.stderr)

    with csv_file.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated status for '{tiff_name}' ({year}) to '{new_status}'.", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage TIFF processing status via a CSV file.")
    parser.add_argument('--csv', required=True, help="Path to the status CSV file.")
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # 'init' command
    parser_init = subparsers.add_parser('init', help="Initialize the status CSV file.")
    parser_init.add_argument('--source-dir', required=True, help="Directory containing the source TIFF files.")
    parser_init.add_argument('--years', required=True, nargs='+', help="List of years to process.")
    
    # 'claim' command
    parser_claim = subparsers.add_parser('claim', help="Claim an available task for processing.")
    
    # 'update' command
    parser_update = subparsers.add_parser('update', help="Update the status of a task.")
    parser_update.add_argument('--name', required=True, help="The tiff_name of the task.")
    parser_update.add_argument('--year', required=True, help="The year of the task.")
    parser_update.add_argument('--status', required=True, choices=[STATUS_PROCESSED, STATUS_BROKEN], help="The new status of the task.")

    args = parser.parse_args()
    
    if args.command == 'init':
        init_csv(args.csv, args.source_dir, args.years)
    elif args.command == 'claim':
        claim_task(args.csv)
    elif args.command == 'update':
        update_status(args.csv, args.name, args.year, args.status)