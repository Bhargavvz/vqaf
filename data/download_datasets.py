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
    Download VQA-RAD dataset from HuggingFace.
    
    Uses the flaviagiammarino/vqa-rad HuggingFace dataset which contains
    radiology images with question-answer pairs (~3,515 QA pairs).
    
    Args:
        output_dir: Directory to save the dataset.
    """
    logger.info("=" * 60)
    logger.info("VQA-RAD Dataset Download (via HuggingFace)")
    logger.info("=" * 60)
    
    try:
        from datasets import load_dataset
        
        cache_dir = Path(output_dir) / "vqa_rad_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading VQA-RAD from HuggingFace Hub...")
        dataset = load_dataset(
            "flaviagiammarino/vqa-rad",
            cache_dir=str(cache_dir),
        )
        
        logger.info("VQA-RAD downloaded successfully!")
        for split_name, split_data in dataset.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")
        
        # Also save as JSON for our dataset loader
        vqa_rad_dir = Path(output_dir) / "vqa_rad"
        vqa_rad_dir.mkdir(parents=True, exist_ok=True)
        images_dir = vqa_rad_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        all_samples = []
        img_idx = 0
        for split_name, split_data in dataset.items():
            for item in split_data:
                # Save image to disk
                image = item["image"]
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                img_filename = f"vqarad_{img_idx:05d}.jpg"
                image.save(str(images_dir / img_filename))
                
                all_samples.append({
                    "image_name": img_filename,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "answer_type": "CLOSED" if str(item.get("answer", "")).lower() in ["yes", "no"] else "OPEN",
                    "question_type": "general",
                })
                img_idx += 1
        
        # Save JSON
        json_file = vqa_rad_dir / "VQA_RAD Dataset Public.json"
        with open(json_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        logger.info(f"VQA-RAD saved: {len(all_samples)} samples, {img_idx} images to {vqa_rad_dir}")
        
    except ImportError:
        logger.error("'datasets' library not installed. Run: pip install datasets")
    except Exception as e:
        logger.error(f"Failed to download VQA-RAD: {e}")
        logger.warning("Creating placeholder VQA-RAD data for development...")
        vqa_rad_dir = Path(output_dir) / "vqa_rad"
        vqa_rad_dir.mkdir(parents=True, exist_ok=True)
        (vqa_rad_dir / "images").mkdir(exist_ok=True)
        _create_placeholder_vqa_rad(vqa_rad_dir)


def download_path_vqa(output_dir: str):
    """
    Download PathVQA dataset and save as JSON + images to disk.
    
    Saves in the same format as VQA-RAD for consistent loading.
    
    Args:
        output_dir: Directory to save the dataset.
    """
    logger.info("=" * 60)
    logger.info("PathVQA Dataset Download (via HuggingFace)")
    logger.info("=" * 60)
    
    path_vqa_dir = Path(output_dir) / "path_vqa"
    
    # Check if already saved
    if (path_vqa_dir / "train.json").exists():
        logger.info(f"PathVQA already saved at {path_vqa_dir}")
        return
    
    try:
        from datasets import load_dataset
        
        cache_dir = Path(output_dir) / "path_vqa_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading PathVQA from HuggingFace Hub...")
        dataset = load_dataset(
            "flaviagiammarino/path-vqa",
            cache_dir=str(cache_dir),
        )
        
        logger.info("PathVQA downloaded successfully!")
        
        # Save each split as JSON + images
        split_map = {"train": "train", "validation": "val", "test": "test"}
        
        for hf_split, our_split in split_map.items():
            if hf_split not in dataset:
                continue
            
            split_data = dataset[hf_split]
            images_dir = path_vqa_dir / our_split / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            samples = []
            for i, item in enumerate(split_data):
                # Save image
                image = item["image"]
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                img_filename = f"pathvqa_{i:05d}.jpg"
                image.save(str(images_dir / img_filename))
                
                answer = str(item.get("answer", ""))
                samples.append({
                    "image_name": img_filename,
                    "question": str(item.get("question", "")),
                    "answer": answer,
                    "answer_type": "closed" if answer.lower() in ["yes", "no"] else "open",
                    "question_type": "pathology",
                })
            
            # Save JSON
            json_file = path_vqa_dir / f"{our_split}.json"
            with open(json_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            logger.info(f"  {our_split}: saved {len(samples)} samples to {path_vqa_dir / our_split}")
        
        logger.info(f"PathVQA saved to disk: {path_vqa_dir}")
        
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
