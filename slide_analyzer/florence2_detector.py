"""Florence-2 detector for diagram-aware object detection and OCR."""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from .config import DEVICE, FLORENCE2_MODEL, FLORENCE2_OCR_THRESHOLD, FLORENCE2_FORCE_CPU


class Florence2Detector:
    """Florence-2 based detector for diagrams and technical content."""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cpu"  # Default to CPU, updated in load()
        
    def load(self):
        """Load Florence-2 model and processor."""
        print(f"Loading Florence-2 ({FLORENCE2_MODEL})...")
        
        # Determine device (use CPU if forced or if AMD GPU has issues)
        device = "cpu" if FLORENCE2_FORCE_CPU else DEVICE
        
        self.processor = AutoProcessor.from_pretrained(
            FLORENCE2_MODEL, 
            trust_remote_code=True
        )
        
        # Load with proper dtype and device to avoid fp16/caching issues
        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",  # Fix for _supports_sdpa error
        }
        
        # Only use float16 for NVIDIA GPUs, use float32 for CPU or AMD
        if device == "cuda" and not FLORENCE2_FORCE_CPU:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = device
        else:
            # CPU or AMD - use float32 for stability
            model_kwargs["torch_dtype"] = torch.float32
            print(f"  Using CPU mode (device={device})")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            FLORENCE2_MODEL,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if "device_map" not in model_kwargs:
            self.model = self.model.to(device)
        
        # Disable KV caching to avoid ROCm/CPU issues
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self.device = device
        print("✓ Florence-2 loaded")
    
    def run_task(self, image_pil: Image.Image, task: str, text_input=None):
        """
        Run a Florence-2 task.
        
        Args:
            image_pil: PIL Image
            task: Task name (e.g., '<OCR_WITH_REGION>', '<CAPTION_TO_PHRASE_GROUNDING>')
            text_input: Optional text input for grounding tasks
            
        Returns:
            dict: Task results
        """
        if text_input:
            prompt = task + text_input
        else:
            prompt = task
        
        inputs = self.processor(
            text=prompt, 
            images=image_pil, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Use greedy decoding and disable cache to avoid ROCm issues
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=1,  # Greedy decoding to avoid cache bug
                do_sample=False,
                use_cache=False
            )
        
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]
        
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image_pil.width, image_pil.height)
        )
        
        return parsed.get(task, {})
    
    def extract_ocr_with_boxes(self, image_pil: Image.Image):
        """
        Extract text with bounding boxes using Florence-2 OCR.
        
        Returns:
            list: List of dicts with 'text', 'box', 'confidence'
        """
        result = self.run_task(image_pil, '<OCR_WITH_REGION>')
        
        anchors = []
        quad_boxes = result.get('quad_boxes', [])
        labels = result.get('labels', [])
        
        for text, box in zip(labels, quad_boxes):
            if not text or not text.strip():
                continue
            
            # Convert quad box to [x1, y1, x2, y2]
            xs = [box[i] for i in range(0, len(box), 2)]
            ys = [box[i] for i in range(1, len(box), 2)]
            
            anchors.append({
                'text': text.strip(),
                'box': [min(xs), min(ys), max(xs), max(ys)],
                'confidence': 1.0,  # Florence-2 doesn't provide confidence
                'source': 'florence2'
            })
        
        return anchors
    
    def detect_objects_with_caption(self, image_pil: Image.Image):
        """
        Detect objects with captions using Florence-2.
        
        Returns:
            list: List of dicts with 'label', 'box', 'score'
        """
        result = self.run_task(image_pil, '<DENSE_REGION_CAPTION>')
        
        detections = []
        bboxes = result.get('bboxes', [])
        labels = result.get('labels', [])
        
        for label, box in zip(labels, bboxes):
            detections.append({
                'label': label,
                'box': box,
                'score': 1.0  # Florence-2 doesn't provide scores
            })
        
        return detections
    
    def phrase_grounding(self, image_pil: Image.Image, text_phrase: str):
        """
        Ground a text phrase to image regions.
        
        Args:
            image_pil: PIL Image
            text_phrase: Text to ground (e.g., "blue box labeled 'Mesos master'")
            
        Returns:
            list: List of boxes [x1, y1, x2, y2] or empty if not found
        """
        result = self.run_task(
            image_pil, 
            '<CAPTION_TO_PHRASE_GROUNDING>',
            text_input=text_phrase
        )
        
        bboxes = result.get('bboxes', [])
        labels = result.get('labels', [])
        
        # Return boxes that match the phrase
        matched_boxes = []
        for label, box in zip(labels, bboxes):
            # Florence-2 returns the grounded regions
            matched_boxes.append(box)
        
        return matched_boxes


def load_florence2_detector():
    """Load and return Florence-2 detector instance."""
    detector = Florence2Detector()
    detector.load()
    return detector

