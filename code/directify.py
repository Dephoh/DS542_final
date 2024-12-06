import os
import shutil
from pathlib import Path

def directify(file_paths):

    output_dir = Path("selected")
    output_dir.mkdir(exist_ok=True)
    successful = []
    failed = []
    
    for file_path in file_paths:
        try:
            source = Path(file_path)

            if not source.exists():
                failed.append((file_path, "File not found"))
                continue
                
            dest = output_dir / source.name
            shutil.copy2(source, dest)
            successful.append(file_path)
            
        except Exception as e:
            failed.append((file_path, str(e)))
    
    print(f"\nCopied {len(successful)} files successfully to {output_dir}")
    
    if failed:
        print("\nFailed to copy the following files:")
        for path, error in failed:
            print(f"- {path}: {error}")
