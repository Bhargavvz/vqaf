"""
Dataset Download Script
========================
Downloads VQA-RAD and PathVQA datasets for Medical VQA training.

Usage:
    python -m medical_vqa.data.download_datasets --output_dir ./data
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm


def download_file(url: str, dest_path: str, desc: str = "Downloading") -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from.
        dest_path: Local destination file path.
        desc: Description for progress bar.
    
    Returns:
        True if download succeeded.
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(dest_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=desc
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_vqa_rad(output_dir: str):
    """
    Download VQA-RAD dataset.
    
    The VQA-RAD dataset is available from the original authors.
    This function attempts to download from known public mirrors.
    
    Args:
        output_dir: Directory to save the dataset.
    """
    vqa_rad_dir = Path(output_dir) / "vqa_rad"
    vqa_rad_dir.mkdir(parents=True, exist_ok=True)
    images_dir = vqa_rad_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("VQA-RAD Dataset Download")
    logger.info("=" * 60)
    
    # VQA-RAD is typically distributed as a JSON + image folder
    # Primary source: https://osf.io/89kps/ (Open Science Framework)
    vqa_rad_urls = [
        # OSF download link for VQA-RAD
        "https://osf.io/89kps/download",
    ]
    
    json_file = vqa_rad_dir / "VQA_RAD Dataset Public.json"
    
    if json_file.exists():
        logger.info(f"VQA-RAD JSON already exists at {json_file}")
    else:
        logger.info(
            "VQA-RAD requires manual download due to licensing.\n"
            "Please follow these steps:\n"
            "1. Visit: https://osf.io/89kps/\n"
            "2. Download 'VQA_RAD Dataset Public.json'\n"
            "3. Download the image folder\n"
            f"4. Place files in: {vqa_rad_dir}\n"
            "   - JSON file directly in the folder\n"
            "   - Images in the 'images/' subfolder\n"
        )
        
        # Try automated download
        for url in vqa_rad_urls:
            zip_path = vqa_rad_dir / "vqa_rad.zip"
            logger.info(f"Attempting download from {url}...")
            if download_file(url, str(zip_path), "VQA-RAD"):
                try:
                    with zipfile.ZipFile(str(zip_path), 'r') as zf:
                        zf.extractall(str(vqa_rad_dir))
                    logger.info("VQA-RAD extracted successfully!")
                    zip_path.unlink()  # Remove zip
                    break
                except zipfile.BadZipFile:
                    logger.warning("Downloaded file is not a valid ZIP. Manual download required.")
                    zip_path.unlink()
    
    # Create a sample/placeholder if download failed
    if not json_file.exists():
        logger.warning("Creating placeholder VQA-RAD data for development...")
        _create_placeholder_vqa_rad(vqa_rad_dir)


def download_path_vqa(output_dir: str):
    """
    Download PathVQA dataset using HuggingFace datasets.
    
    Args:
        output_dir: Directory for caching.
    """
    logger.info("=" * 60)
    logger.info("PathVQA Dataset Download (via HuggingFace)")
    logger.info("=" * 60)
    
    try:
        from datasets import load_dataset
        
        cache_dir = Path(output_dir) / "path_vqa_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading PathVQA from HuggingFace Hub...")
        dataset = load_dataset(
            "flaviagiammarino/path-vqa",
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        
        logger.info(f"PathVQA downloaded successfully!")
        for split_name, split_data in dataset.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")
        
    except ImportError:
        logger.error("'datasets' library not installed. Run: pip install datasets")
    except Exception as e:
        logger.error(f"Failed to download PathVQA: {e}")


def _create_placeholder_vqa_rad(output_dir: Path):
    """Create placeholder VQA-RAD data for development/testing."""
    placeholder_data = [
        {
            "image_name": "placeholder_xray.jpg",
            "question": "Is there cardiomegaly?",
            "answer": "no",
            "answer_type": "CLOSED",
            "question_type": "Abnormality"
        },
        {
            "image_name": "placeholder_xray.jpg",
            "question": "What organ system is shown in this image?",
            "answer": "chest",
            "answer_type": "OPEN",
            "question_type": "Organ"
        },
        {
            "image_name": "placeholder_xray.jpg",
            "question": "Is there pleural effusion?",
            "answer": "yes",
            "answer_type": "CLOSED",
            "question_type": "Abnormality"
        },
        {
            "image_name": "placeholder_xray.jpg",
            "question": "What type of imaging modality is used?",
            "answer": "x-ray",
            "answer_type": "OPEN",
            "question_type": "Modality"
        },
        {
            "image_name": "placeholder_xray.jpg",
            "question": "How many ribs are visible?",
            "answer": "12",
            "answer_type": "OPEN",
            "question_type": "Counting"
        },
    ]
    
    json_file = output_dir / "VQA_RAD Dataset Public.json"
    with open(json_file, 'w') as f:
        json.dump(placeholder_data, f, indent=2)
    
    # Create a placeholder image
    from PIL import Image
    img = Image.new("RGB", (448, 448), (128, 128, 128))
    img.save(str(output_dir / "images" / "placeholder_xray.jpg"))
    
    logger.info(f"Created placeholder data with {len(placeholder_data)} samples")


def main():
    parser = argparse.ArgumentParser(description="Download Medical VQA Datasets")
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Directory to save datasets"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["vqa_rad", "path_vqa"],
        choices=["vqa_rad", "path_vqa"],
        help="Which datasets to download"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if "vqa_rad" in args.datasets:
        download_vqa_rad(args.output_dir)
    
    if "path_vqa" in args.datasets:
        download_path_vqa(args.output_dir)
    
    logger.info("Dataset download complete!")


if __name__ == "__main__":
    main()
