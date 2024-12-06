import os
import shutil
from pathlib import Path

def directify(file_paths):
    """
    Copy images from given file paths to a 'selected' directory.
    
    Args:
        file_paths (list): List of strings containing file paths to images
    """
    # Create the selected directory if it doesn't exist
    output_dir = Path("selected")
    output_dir.mkdir(exist_ok=True)
    
    # Keep track of successful and failed copies
    successful = []
    failed = []
    
    for file_path in file_paths:
        try:
            # Convert string path to Path object
            source = Path(file_path)
            
            # Check if source file exists
            if not source.exists():
                failed.append((file_path, "File not found"))
                continue
                
            # Create destination path
            dest = output_dir / source.name
            
            # Copy the file
            shutil.copy2(source, dest)
            successful.append(file_path)
            
        except Exception as e:
            failed.append((file_path, str(e)))
    
    # Print summary
    print(f"\nCopied {len(successful)} files successfully to {output_dir}")
    
    if failed:
        print("\nFailed to copy the following files:")
        for path, error in failed:
            print(f"- {path}: {error}")
