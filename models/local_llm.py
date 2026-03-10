import os
import anyio
from pathlib import Path
from typing import Optional

# Try to import mlx-lm; provide a fallback for non-Apple Silicon systems
try:
    from mlx_lm import load, generate, sample_utils
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from config.config import settings
from utils.logger import logger

try:
    import psutil
except ImportError:
    logger.warning("psutil not found. Memory checks will be skipped.")
    psutil = None

class LocalLLM:
    """
    Local LLM wrapper for generating responses using 'mlx-lm'.
    Falls back to a 'SIMULATED' mode if mlx-lm or sufficient hardware isn't available.
    """
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = (os.environ.get("VERISCAN_LLM_MODEL") or model_id or settings.LLM_MODEL_ID).strip()
        self.model = None
        self.tokenizer = None
        self.is_simulated = False

        if not HAS_MLX:
            logger.warning(f"MLX-LM not found. LocalLLM ({self.model_id}) will operate in SIMULATED mode.")
            self.is_simulated = True
            return

        logger.info(f"Loading local MLX model: {self.model_id}...")
        
        # Resource check for larger models
        if "8B" in self.model_id.upper() and not self._check_memory():
            logger.warning(f"Insufficient RAM for 8B model. Switching to SIMULATED mode for stability.")
            self.is_simulated = True
            return

        try:
            # MLX-LM handles downloading and loading automatically
            self.model, self.tokenizer = load(self.model_id)
            logger.info("Local MLX LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}. Falling back to SIMULATED mode.")
            self.is_simulated = True

    def _check_memory(self) -> bool:
        """Verify if the system has enough free RAM for the selected model."""
        if psutil is None:
            logger.warning("psutil not available, skipping memory check.")
            return True
            
        try:
            free_gb = psutil.virtual_memory().available / (1024 ** 3)
            logger.info(f"System Memory: {free_gb:.1f}GB available.")
            return free_gb >= settings.MEMORY_THRESHOLD_GB
        except ImportError:
            # If psutil is missing, we assume a typical laptop might struggle with 8B
            # but we allow it if the user hasn't explicitly set a high threshold
            return True

    def generate(self, prompt: str, max_tokens: int = None, temp: float = None) -> str:
        """Generate response from the local MLX model or a simulation stub."""
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temp = temp if temp is not None else settings.LLM_TEMPERATURE

        if not HAS_MLX or self.model is None:
            return f"[SIMULATED RESPONSE for {self.model_id}]: Based on the security logs, this transaction appears suspicious due to a geographic anomaly."

        # Simple Instruct template for Llama-3
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        sampler = sample_utils.make_sampler(temp=temp)
        
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )
        # Strip Llama-3 special tokens if they leak
        for token in ["<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "assistant", "user"]:
            response = response.replace(token, "")

        response = response.strip()
        
        # Ensure response ends at the last complete sentence
        if response and not response.endswith((".", "!", "?")):
            last_punct = max(response.rfind("."), response.rfind("!"), response.rfind("?"))
            if last_punct != -1:
                response = response[:last_punct + 1]

        return response

    async def generate_async(self, prompt: str, max_tokens: int = None, temp: float = None) -> str:
        """Run generation in a separate thread to avoid blocking the event loop."""
        return await anyio.to_thread.run_sync(self.generate, prompt, max_tokens, temp)

if __name__ == "__main__":
    llm = LocalLLM()
    question = "What are the common indicators of credit card fraud?"
    print(f"\nQuestion: {question}")
    response = llm.generate(question)
    print(f"\nResponse: {response}")
