"""
Medical VQA Model Module
=========================
Qwen3-VL-8B Vision-Language Model with QLoRA fine-tuning
for Medical Visual Question Answering.

Features:
- 4-bit quantization (QLoRA) for memory efficiency
- LoRA adapters on attention projections
- Flash Attention 2 for speed
- BF16 mixed precision
- Knowledge-guided prompt construction
"""

import os
from typing import Dict, List, Optional, Any, Tuple

import torch
from loguru import logger


class MedicalVQAModel:
    """
    Medical VQA model wrapper around Qwen3-VL-8B.
    
    Handles:
        - Model loading with quantization and LoRA
        - Prompt construction with knowledge injection
        - Inference with generation parameters
        - Gradient checkpointing for training
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full configuration dictionary with model, quantization,
                    and LoRA settings.
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.quant_config = config.get("quantization", {})
        self.lora_config = config.get("lora", {})
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = None
        
        self.model_name = self.model_config.get("name", "Qwen/Qwen3-VL-8B")
        
        logger.info(f"MedicalVQAModel configured with: {self.model_name}")
    
    def load_model(self, for_training: bool = False) -> "MedicalVQAModel":
        """
        Load the Qwen3-VL model with quantization and LoRA.
        
        Args:
            for_training: If True, enables gradient checkpointing
                         and prepares for fine-tuning.
        
        Returns:
            Self for chaining.
        """
        from transformers import (
            Qwen3VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
            TaskType,
        )
        
        logger.info(f"Loading model: {self.model_name}")
        
        # --- Quantization Config ---
        bnb_config = None
        if self.quant_config.get("enabled", True):
            compute_dtype = getattr(
                torch,
                self.quant_config.get("compute_dtype", "bfloat16")
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.quant_config.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.quant_config.get("use_double_quant", True),
            )
            logger.info("4-bit quantization enabled (NF4 + double quant)")
        
        # --- Determine torch dtype ---
        torch_dtype_str = self.model_config.get("torch_dtype", "bfloat16")
        torch_dtype = getattr(torch, torch_dtype_str, torch.bfloat16)
        
        # --- Load Model ---
        model_kwargs = {
            "dtype": torch_dtype,
            "device_map": self.model_config.get("device_map", "auto"),
            "trust_remote_code": self.model_config.get("trust_remote_code", True),
        }
        
        # Add attention implementation
        attn_impl = self.model_config.get("attn_implementation")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        
        # Add quantization config
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Disable cache for training (required for gradient checkpointing)
        if for_training and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        
        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        
        # --- Prepare for Training ---
        if for_training:
            if bnb_config:
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=self.model_config.get(
                        "gradient_checkpointing", True
                    )
                )
                logger.info("Model prepared for k-bit training")
            
            # Apply LoRA
            self._apply_lora()
        
        # --- Load Processor ---
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        
        # Set device
        self.device = next(self.model.parameters()).device
        
        logger.info(f"Model ready on device: {self.device}")
        return self
    
    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            r=self.lora_config.get("r", 16),
            lora_alpha=self.lora_config.get("lora_alpha", 32),
            lora_dropout=self.lora_config.get("lora_dropout", 0.05),
            target_modules=self.lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            bias=self.lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable_params, total_params = self.model.get_nb_trainable_parameters()
        logger.info(
            f"LoRA applied: {trainable_params:,} trainable / "
            f"{total_params:,} total "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
    
    def load_trained_adapter(self, adapter_path: str):
        """
        Load a previously trained LoRA adapter.
        
        Args:
            adapter_path: Path to the saved adapter weights.
        """
        from peft import PeftModel
        
        if self.model is None:
            self.load_model(for_training=False)
        
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            is_trainable=False
        )
        logger.info(f"Loaded trained adapter from: {adapter_path}")
    
    def build_prompt(
        self,
        question: str,
        knowledge: str = "",
        include_explanation: bool = True
    ) -> str:
        """
        Build the knowledge-guided prompt for medical VQA.
        
        Prompt Format:
            [IMAGE]
            Question: {question}
            Relevant Medical Knowledge: {retrieved facts}
            Answer with detailed clinical explanation:
        
        Args:
            question: The medical question.
            knowledge: Retrieved medical knowledge text.
            include_explanation: Whether to request explanation.
        
        Returns:
            Formatted prompt string.
        """
        prompt_parts = [f"Question: {question}"]
        
        if knowledge:
            prompt_parts.append(f"\nRelevant Medical Knowledge:\n{knowledge}")
        
        if include_explanation:
            prompt_parts.append(
                "\nProvide a precise answer followed by a brief clinical explanation "
                "grounded in the medical knowledge provided."
            )
        else:
            prompt_parts.append("\nProvide a precise answer:")
        
        return "\n".join(prompt_parts)
    
    def build_chat_messages(
        self,
        image_path: str,
        question: str,
        knowledge: str = "",
        answer: Optional[str] = None
    ) -> List[Dict]:
        """
        Build Qwen VL chat messages format.
        
        Args:
            image_path: Path to the medical image.
            question: The VQA question.
            knowledge: Retrieved medical knowledge.
            answer: Ground-truth answer (for training). None for inference.
        
        Returns:
            List of message dicts in Qwen chat format.
        """
        # System message
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert medical imaging AI assistant. Analyze medical "
                "images and answer questions with clinical accuracy. Always provide "
                "a precise answer followed by a brief medical explanation."
            )
        }
        
        # User message with image and question
        user_content = []
        
        # Add image
        if image_path and os.path.exists(str(image_path)):
            user_content.append({
                "type": "image",
                "image": f"file://{image_path}"
            })
        
        # Add text prompt
        prompt_text = self.build_prompt(question, knowledge)
        user_content.append({
            "type": "text",
            "text": prompt_text
        })
        
        user_msg = {
            "role": "user",
            "content": user_content
        }
        
        messages = [system_msg, user_msg]
        
        # Add assistant response for training
        if answer is not None:
            assistant_msg = {
                "role": "assistant",
                "content": answer
            }
            messages.append(assistant_msg)
        
        return messages
    
    @torch.inference_mode()
    def generate(
        self,
        image,
        question: str,
        knowledge: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Generate answer for a medical VQA query.
        
        Args:
            image: PIL Image or path to image.
            question: Medical question.
            knowledge: Retrieved medical knowledge.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
        
        Returns:
            Dict with 'answer', 'full_response', and 'input_tokens'.
        """
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info
        
        # Handle image input
        if isinstance(image, str):
            image_source = f"file://{image}"
        elif isinstance(image, PILImage.Image):
            # Save temporarily for Qwen processing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                image_source = f"file://{tmp.name}"
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert medical imaging AI. Provide precise, "
                    "clinically accurate answers with brief explanations."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_source},
                    {"type": "text", "text": self.build_prompt(question, knowledge)}
                ]
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        
        # Decode (only new tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        # Parse answer and explanation
        answer, explanation = self._parse_response(response)
        
        return {
            "answer": answer,
            "explanation": explanation,
            "full_response": response,
            "input_tokens": input_len,
            "output_tokens": len(generated_ids[0])
        }
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the model response into answer and explanation.
        
        Args:
            response: Raw model response text.
        
        Returns:
            Tuple of (answer, explanation).
        """
        response = response.strip()
        
        # Try to split at common delimiters
        for delimiter in [
            "\nExplanation:", "\nexplanation:",
            "\nReason:", "\nreason:",
            "\nBecause", "\nbecause",
            ". This", ". The",
        ]:
            if delimiter in response:
                parts = response.split(delimiter, 1)
                answer = parts[0].strip()
                explanation = (delimiter.strip(": \n") + " " + parts[1]).strip()
                return answer, explanation
        
        # If response is short, it's likely just an answer
        if len(response.split()) <= 5:
            return response, ""
        
        # First sentence is answer, rest is explanation
        sentences = response.split(". ", 1)
        if len(sentences) > 1:
            return sentences[0].strip(), sentences[1].strip()
        
        return response, ""
    
    def save_adapter(self, output_dir: str):
        """Save the LoRA adapter weights."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        logger.info(f"Adapter saved to: {output_dir}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information summary."""
        info = {
            "model_name": self.model_name,
            "quantization": self.quant_config.get("enabled", False),
            "lora_rank": self.lora_config.get("r", 16),
            "lora_alpha": self.lora_config.get("lora_alpha", 32),
        }
        
        if self.model is not None:
            info["total_params"] = self.model.num_parameters()
            try:
                trainable, total = self.model.get_nb_trainable_parameters()
                info["trainable_params"] = trainable
                info["trainable_pct"] = f"{100 * trainable / total:.2f}%"
            except Exception:
                pass
        
        return info
