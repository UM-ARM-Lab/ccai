#!/usr/bin/env python3
"""
Utility script for managing PyTorch compilation cache.

This script provides utilities to:
- List cache contents and sizes
- Clean up old or unused cache files
- Monitor cache usage over time
- Set up cache directories

Usage:
    python manage_compilation_cache.py --list
    python manage_compilation_cache.py --clean
    python manage_compilation_cache.py --clean-old --days 7
    python manage_compilation_cache.py --stats
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def get_cache_directory():
    """Get the compilation cache directory."""
    ccai_path = Path(__file__).parent.parent
    return ccai_path / "compiled_models_cache"


def format_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def list_cache_contents(cache_dir):
    """List all files in the cache directory."""
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    cache_files = list(cache_dir.glob("*.pt"))
    if not cache_files:
        print("No cache files found.")
        return
    
    print(f"\nCache directory: {cache_dir}")
    print(f"Total files: {len(cache_files)}")
    print("\nCache files:")
    print("-" * 80)
    print(f"{'Filename':<50} {'Size':<10} {'Modified':<20}")
    print("-" * 80)
    
    total_size = 0
    for cache_file in sorted(cache_files):
        stat = cache_file.stat()
        size = stat.st_size
        total_size += size
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        print(f"{cache_file.name:<50} {format_size(size):<10} {modified.strftime('%Y-%m-%d %H:%M'):<20}")
    
    print("-" * 80)
    print(f"Total cache size: {format_size(total_size)}")


def clean_all_cache(cache_dir):
    """Remove all cache files."""
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    cache_files = list(cache_dir.glob("*.pt"))
    if not cache_files:
        print("No cache files to clean.")
        return
    
    total_size = sum(f.stat().st_size for f in cache_files)
    
    answer = input(f"Delete {len(cache_files)} cache files ({format_size(total_size)})? [y/N]: ")
    if answer.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    deleted_count = 0
    deleted_size = 0
    
    for cache_file in cache_files:
        try:
            size = cache_file.stat().st_size
            cache_file.unlink()
            deleted_count += 1
            deleted_size += size
            print(f"Deleted: {cache_file.name}")
        except Exception as e:
            print(f"Error deleting {cache_file.name}: {e}")
    
    print(f"\nCleaned up {deleted_count} files, freed {format_size(deleted_size)}")


def clean_old_cache(cache_dir, days):
    """Remove cache files older than specified days."""
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    cutoff_date = datetime.now() - timedelta(days=days)
    cache_files = list(cache_dir.glob("*.pt"))
    old_files = []
    
    for cache_file in cache_files:
        modified = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if modified < cutoff_date:
            old_files.append(cache_file)
    
    if not old_files:
        print(f"No cache files older than {days} days found.")
        return
    
    total_size = sum(f.stat().st_size for f in old_files)
    print(f"Found {len(old_files)} files older than {days} days ({format_size(total_size)})")
    
    answer = input(f"Delete these old cache files? [y/N]: ")
    if answer.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    deleted_count = 0
    deleted_size = 0
    
    for cache_file in old_files:
        try:
            size = cache_file.stat().st_size
            cache_file.unlink()
            deleted_count += 1
            deleted_size += size
            print(f"Deleted: {cache_file.name} (modified: {datetime.fromtimestamp(cache_file.stat().st_mtime)})")
        except Exception as e:
            print(f"Error deleting {cache_file.name}: {e}")
    
    print(f"\nCleaned up {deleted_count} old files, freed {format_size(deleted_size)}")


def show_cache_stats(cache_dir):
    """Show detailed cache statistics."""
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    cache_files = list(cache_dir.glob("*.pt"))
    if not cache_files:
        print("No cache files found.")
        return
    
    # Analyze files by model type
    model_stats = {}
    total_size = 0
    oldest_file = None
    newest_file = None
    
    for cache_file in cache_files:
        # Extract model name from filename
        name_parts = cache_file.name.split('_')
        if len(name_parts) >= 2:
            model_name = name_parts[0]
        else:
            model_name = "Unknown"
        
        stat = cache_file.stat()
        size = stat.st_size
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        if model_name not in model_stats:
            model_stats[model_name] = {'count': 0, 'size': 0, 'oldest': modified, 'newest': modified}
        
        model_stats[model_name]['count'] += 1
        model_stats[model_name]['size'] += size
        model_stats[model_name]['oldest'] = min(model_stats[model_name]['oldest'], modified)
        model_stats[model_name]['newest'] = max(model_stats[model_name]['newest'], modified)
        
        total_size += size
        
        if oldest_file is None or modified < oldest_file[1]:
            oldest_file = (cache_file, modified)
        if newest_file is None or modified > newest_file[1]:
            newest_file = (cache_file, modified)
    
    print(f"\nCache Statistics")
    print("=" * 60)
    print(f"Cache directory: {cache_dir}")
    print(f"Total files: {len(cache_files)}")
    print(f"Total size: {format_size(total_size)}")
    print(f"Average file size: {format_size(total_size / len(cache_files))}")
    
    if oldest_file:
        print(f"Oldest file: {oldest_file[0].name} ({oldest_file[1].strftime('%Y-%m-%d %H:%M')})")
    if newest_file:
        print(f"Newest file: {newest_file[0].name} ({newest_file[1].strftime('%Y-%m-%d %H:%M')})")
    
    print(f"\nBy Model Type:")
    print("-" * 60)
    print(f"{'Model':<20} {'Files':<8} {'Size':<12} {'Age Range':<20}")
    print("-" * 60)
    
    for model_name, stats in sorted(model_stats.items()):
        age_range = f"{stats['oldest'].strftime('%m/%d')} - {stats['newest'].strftime('%m/%d')}"
        print(f"{model_name:<20} {stats['count']:<8} {format_size(stats['size']):<12} {age_range:<20}")


def setup_cache_directory():
    """Create cache directory if it doesn't exist."""
    cache_dir = get_cache_directory()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory set up at: {cache_dir}")
    
    # Create a .gitignore file to exclude cache from git
    gitignore_path = cache_dir / ".gitignore"
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write("# Ignore all compiled model cache files\n")
            f.write("*.pt\n")
            f.write("*.pkl\n")
        print("Created .gitignore for cache directory")


def main():
    parser = argparse.ArgumentParser(
        description="Manage PyTorch model compilation cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    # List all cache files
  %(prog)s --clean                   # Clean all cache files
  %(prog)s --clean-old --days 7      # Clean files older than 7 days
  %(prog)s --stats                   # Show cache statistics
  %(prog)s --setup                   # Create cache directory
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List all cache files')
    parser.add_argument('--clean', action='store_true',
                       help='Remove all cache files')
    parser.add_argument('--clean-old', action='store_true',
                       help='Remove old cache files')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days for --clean-old (default: 7)')
    parser.add_argument('--stats', action='store_true',
                       help='Show detailed cache statistics')
    parser.add_argument('--setup', action='store_true',
                       help='Set up cache directory')
    
    args = parser.parse_args()
    
    if not any([args.list, args.clean, args.clean_old, args.stats, args.setup]):
        parser.print_help()
        return
    
    cache_dir = get_cache_directory()
    
    if args.setup:
        setup_cache_directory()
    
    if args.list:
        list_cache_contents(cache_dir)
    
    if args.clean:
        clean_all_cache(cache_dir)
    
    if args.clean_old:
        clean_old_cache(cache_dir, args.days)
    
    if args.stats:
        show_cache_stats(cache_dir)


if __name__ == "__main__":
    main() 