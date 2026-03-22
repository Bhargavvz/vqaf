"""
Input Processor Module
=======================
Handles multimodal input processing for Qwen3-VL model.
Converts image + text inputs into model-ready tensors.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from loguru import logger


class MedicalVQAProcessor:
    """
    Processor for preparing multimodal inputs for the Medical VQA model.
    
    Handles:
        - Image loading and preprocessing
        - Chat template application
        - Tokenization of text inputs
        - Batch collation for training
    """
    
    def __init__(self, processor, max_seq_length: int = 2048):
        """
        Args:
            processor: HuggingFace AutoProcessor (Qwen VL processor).
            max_seq_length: Maximum sequence length for tokenization.
        """
        self.processor = processor
        self.max_seq_length = max_seq_length
    
    def process_sample(
        self,
        image,
        question: str,
        knowledge: str = "",
        answer: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single VQA sample into model inputs.
        
        Args:
            image: PIL Image or image file path.
            question: Medical question text.
            knowledge: Retrieved medical knowledge.
            answer: Ground-truth answer (for training).
            system_prompt: Optional system prompt override.
        
        Returns:
            Dictionary of tensor inputs for the model.
        """
        from qwen_vl_utils import process_vision_info
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are an expert medical imaging AI assistant. Analyze medical "
                "images and answer questions with clinical accuracy. Provide a "
                "precise answer followed by a brief medical explanation."
            )
        
        # Handle image — resize to limit token count
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Resize to a consistent resolution to avoid variable token lengths
        pil_image = pil_image.resize((448, 448), Image.BILINEAR)
        
        # Save to temp file for qwen_vl_utils
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp.name)
            image_source = f"file://{tmp.name}"
        
        # Build prompt text
        prompt_parts = [f"Question: {question}"]
        if knowledge:
            prompt_parts.append(f"\nRelevant Medical Knowledge:\n{knowledge}")
        prompt_parts.append(
            "\nProvide a precise answer followed by a brief clinical explanation."
        )
        prompt_text = "\n".join(prompt_parts)
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_source},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        if answer is not None:
            messages.append({
                "role": "assistant",
                "content": answer
            })
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(answer is None)
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Tokenize — NO truncation to avoid image token mismatch
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt"
        )
        
        # Squeeze batch dimension for single sample
        result = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # If sequence is too long, truncate text tokens only (keep image tokens)
        if result["input_ids"].shape[0] > self.max_seq_length:
            result["input_ids"] = result["input_ids"][:self.max_seq_length]
            if "attention_mask" in result:
                result["attention_mask"] = result["attention_mask"][:self.max_seq_length]
        
        # Create labels for training (mask input tokens, only compute loss on output)
        if answer is not None:
            labels = result["input_ids"].clone()
            # Mask all non-answer tokens with -100
            labels = self._mask_non_answer_tokens(labels, text, answer)
            result["labels"] = labels
        
        # Clean up temp file
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        
        return result
    
    def _mask_non_answer_tokens(
        self,
        labels: torch.Tensor,
        full_text: str,
        answer: str
    ) -> torch.Tensor:
        """
        Mask non-answer tokens in labels with -100 for loss computation.
        Only the answer part should contribute to the loss.
        """
        tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        
        # Encode just the answer to find its length
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        answer_len = len(answer_tokens)
        
        # Mask everything except the last answer_len tokens (+ some margin)
        if answer_len > 0 and answer_len < len(labels):
            mask_end = len(labels) - answer_len - 1  # -1 for EOS
            labels[:mask_end] = -100
        
        # Also mask padding tokens
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        
        return labels
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.
        
        Handles variable-length sequences AND variable-size pixel_values.
        """
        collated = {}
        
        all_keys = batch[0].keys()
        
        for key in all_keys:
            tensors = [sample[key] for sample in batch]
            
            if not isinstance(tensors[0], torch.Tensor):
                collated[key] = tensors
                continue
            
            # Handle pixel_values and image_grid_thw specially
            # These are multi-dimensional and can't be simply padded
            if key in ("pixel_values", "pixel_values_videos"):
                # Concatenate along the first dimension
                collated[key] = torch.cat(tensors, dim=0)
            elif key == "image_grid_thw":
                collated[key] = torch.cat(tensors, dim=0)
            elif tensors[0].dim() == 0:
                # Scalar tensors
                collated[key] = torch.stack(tensors)
            else:
                # 1D tensors (input_ids, attention_mask, labels) — pad to max length
                max_len = max(t.shape[0] for t in tensors)
                
                if key == "labels":
                    pad_value = -100
                elif key == "attention_mask":
                    pad_value = 0
                else:
                    pad_value = 0
                
                padded = []
                for t in tensors:
                    if t.shape[0] < max_len:
                        pad_size = max_len - t.shape[0]
                        t = torch.nn.functional.pad(t, (0, pad_size), value=pad_value)
                    padded.append(t)
                
                collated[key] = torch.stack(padded)
        
        return collated
