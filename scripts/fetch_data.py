"""
Data fetching utilities for the CBRN AI hackathon project.
"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import requests
from tqdm import tqdm


def download_with_progress(url: str, destination: str, description: str = None) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        description: Description for progress bar
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def compute_checksum(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Compute checksum of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use
        
    Returns:
        Hex digest of the file
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def fetch_dataset(name: str, force: bool = False) -> bool:
    """
    Fetch a dataset by name.
    
    Args:
        name: Name of the dataset to fetch
        force: Whether to force re-download
        
    Returns:
        True if successful, False otherwise
    """
    # Mock dataset specification
    spec = {
        "url": f"https://example.com/{name}.tar.gz",
        "checksum": "mock_checksum"
    }
    
    # Create temporary directory for raw data
    raw_dir = Path(tempfile.gettempdir()) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    raw_archive = raw_dir / name
    
    # Check if dataset already exists
    if raw_archive.exists() and not force:
        print(f"Dataset already exists: {raw_archive}")
        print("Use --force to re-download")
        return True
    
    # Download dataset
    print(f"Fetching dataset: {name}")
    success = download_with_progress(spec["url"], str(raw_archive), f"Fetching {name}")
    
    if success:
        print(f"Successfully downloaded {name}")
        return True
    else:
        print(f"Failed to download {name}")
        return False


def list_datasets() -> Dict[str, Any]:
    """
    List available datasets.
    
    Returns:
        Dictionary of available datasets
    """
    # Mock dataset registry
    return {
        "datasets": {
            "test": {
                "url": "https://example.com/test.tar.gz",
                "checksum": "test_checksum"
            },
            "sample": {
                "url": "https://example.com/sample.tar.gz", 
                "checksum": "sample_checksum"
            }
        }
    }


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data fetching utility")
    parser.add_argument("command", choices=["fetch", "list", "checksum"])
    parser.add_argument("--name", help="Dataset name")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--file", help="File path for checksum computation")
    
    args = parser.parse_args()
    
    if args.command == "fetch":
        if not args.name:
            print("Error: --name required for fetch command")
            return 1
        success = fetch_dataset(args.name, force=args.force)
        return 0 if success else 1
    
    elif args.command == "list":
        datasets = list_datasets()
        print(yaml.dump(datasets, default_flow_style=False))
        return 0
    
    elif args.command == "checksum":
        if not args.file:
            print("Error: --file required for checksum command")
            return 1
        checksum = compute_checksum(args.file)
        print(checksum)
        return 0
    
    return 0


if __name__ == "__main__":
    exit(main())
