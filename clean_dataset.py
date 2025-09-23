#!/usr/bin/env python3
"""
Script to clean dataset directories by keeping only specific files.
Enumerates subdirectories one level down and removes all files except the specified ones.
"""

from pathlib import Path


def clean_dataset_directory(base_directory):
    """
    Clean dataset directory by removing all files except specified ones.
    
    Args:
        base_directory (str): Path to the base directory containing subdirectories
    """
    # Files to keep in each subdirectory
    files_to_keep = [
        'cropCell200001.bmp.tif',
        'cropCell200002.bmp.tif',
        'traction.csv',
        'tractionDLTFMs.csv',
        'Cellboundary1.mat'
    ]
    
    base_path = Path(base_directory)
    
    if not base_path.exists():
        print(f"Error: Directory {base_directory} does not exist.")
        return
    
    if not base_path.is_dir():
        print(f"Error: {base_directory} is not a directory.")
        return
    
    # Get all subdirectories one level down
    subdirectories = [item for item in base_path.iterdir() if item.is_dir()]
    
    print(f"Found {len(subdirectories)} subdirectories in {base_directory}")
    
    total_files_removed = 0
    total_files_kept = 0
    
    for subdir in subdirectories:
        print(f"\nProcessing directory: {subdir.name}")
        
        # Get all files in the subdirectory
        files_in_subdir = [item for item in subdir.iterdir() if item.is_file()]
        
        files_removed_in_subdir = 0
        files_kept_in_subdir = 0
        
        for file_path in files_in_subdir:
            filename = file_path.name
            
            if filename in files_to_keep:
                print(f"  Keeping: {filename}")
                files_kept_in_subdir += 1
                total_files_kept += 1
            else:
                try:
                    file_path.unlink()  # Delete the file
                    print(f"  Removed: {filename}")
                    files_removed_in_subdir += 1
                    total_files_removed += 1
                except Exception as e:
                    print(f"  Error removing {filename}: {e}")
        
        print(f"  Summary for {subdir.name}: {files_kept_in_subdir} kept, {files_removed_in_subdir} removed")
    
    print("\nOverall Summary:")
    print(f"Total files kept: {total_files_kept}")
    print(f"Total files removed: {total_files_removed}")
    print(f"Processed {len(subdirectories)} subdirectories")


def preview_cleanup(base_directory):
    """
    Preview what files would be removed without actually deleting them.
    
    Args:
        base_directory (str): Path to the base directory containing subdirectories
    """
    # Files to keep in each subdirectory
    files_to_keep = [
        'cropCell200001.bmp.tif',
        'cropCell200002.bmp.tif',
        'traction.csv',
        'tractionDLTFMs.csv',
        'Cellboundary1.mat'
    ]
    
    base_path = Path(base_directory)
    
    if not base_path.exists():
        print(f"Error: Directory {base_directory} does not exist.")
        return
    
    if not base_path.is_dir():
        print(f"Error: {base_directory} is not a directory.")
        return
    
    # Get all subdirectories one level down
    subdirectories = [item for item in base_path.iterdir() if item.is_dir()]
    
    print(f"PREVIEW MODE - Found {len(subdirectories)} subdirectories in {base_directory}")
    
    total_files_to_remove = 0
    total_files_to_keep = 0
    
    for subdir in subdirectories:
        print(f"\nDirectory: {subdir.name}")
        
        # Get all files in the subdirectory
        files_in_subdir = [item for item in subdir.iterdir() if item.is_file()]
        
        files_to_remove_in_subdir = 0
        files_to_keep_in_subdir = 0
        
        for file_path in files_in_subdir:
            filename = file_path.name
            
            if filename in files_to_keep:
                print(f"  Would keep: {filename}")
                files_to_keep_in_subdir += 1
                total_files_to_keep += 1
            else:
                print(f"  Would remove: {filename}")
                files_to_remove_in_subdir += 1
                total_files_to_remove += 1
        
        print(f"  Summary for {subdir.name}: {files_to_keep_in_subdir} to keep, {files_to_remove_in_subdir} to remove")
    
    print("\nPreview Summary:")
    print(f"Total files to keep: {total_files_to_keep}")
    print(f"Total files to remove: {total_files_to_remove}")
    print(f"Would process {len(subdirectories)} subdirectories")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean dataset directories by keeping only specific files")
    parser.add_argument("directory", help="Base directory containing subdirectories to clean")
    parser.add_argument("--preview", action="store_true", help="Preview what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    if args.preview:
        print("Running in PREVIEW mode - no files will be deleted")
        preview_cleanup(args.directory)
    else:
        print("Running in CLEANUP mode - files will be permanently deleted")
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            clean_dataset_directory(args.directory)
        else:
            print("Operation cancelled.")
