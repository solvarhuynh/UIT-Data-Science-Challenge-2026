"""Local LLM client wrapping the Qwen3 checkpoint via transformers or vLLM."""

import logging
import time
from typing import Generator, Optional

from udsc2026.infrastructure.llm.config import LLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type stubs – transformers / torch are optional at import time so that unit
# tests (which mock this class) do not require GPU or heavy dependencies.
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False


class LLMClient:
    """Load and run a local Qwen3 model for Vietnamese legal question answering.

    Usage::

        config = LLMConfig(model_path="./models/qwen3-legal", device="cuda")
        client = LLMClient(config)
        answer = client.generate("Câu hỏi pháp luật của bạn...")

    The client is *not* thread-safe by default.  TV1 must ensure only one
    concurrent call is made to ``generate`` per ``LLMClient`` instance when
    using the ``transformers`` backend.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialise the LLM client and load model weights from local disk.

        Args:
            config: Runtime configuration including model path, backend and
                inference hyper-parameters.

        Raises:
            ImportError: When ``backend="transformers"`` but the ``transformers``
                package is not installed in the current environment.
            RuntimeError: When the model checkpoint cannot be found at the
                configured path.
        """
        self._config = config
        self._model = None
        self._tokenizer = None
        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load tokenizer and model weights from the local checkpoint path."""
        if self._config.backend == "transformers":
            self._load_transformers()
        else:
            self._load_vllm()

    def _resolve_dtype(self):  # type: ignore[return]
        """Resolve string dtype to the corresponding torch dtype object."""
        if not _TRANSFORMERS_AVAILABLE:
            return None
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping[self._config.dtype]

    def _load_transformers(self) -> None:
        """Load model using the HuggingFace transformers backend."""
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required for the "
                "'transformers' backend.  Install them with: "
                "pip install torch transformers"
            )

        model_path = self._config.model_path
        device = self._config.device
        dtype = self._resolve_dtype()

        logger.info(
            "Loading tokenizer from '%s'…",
            model_path,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[assignment]
            model_path,
            trust_remote_code=True,
        )

        logger.info(
            "Loading model from '%s' (dtype=%s, device=%s)…",
            model_path,
            self._config.dtype,
            device,
        )
        self._model = AutoModelForCausalLM.from_pretrained(  # type: ignore[assignment]
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Model loaded successfully.")

    def _load_vllm(self) -> None:
        """Load model using the vLLM backend for high-throughput serving."""
        try:
            from vllm import LLM  # type: ignore[import]

            self._vllm_engine = LLM(
                model=self._config.model_path,
                dtype=self._config.dtype,
                trust_remote_code=True,
            )
            logger.info("vLLM engine initialised for '%s'.", self._config.model_path)
        except ImportError as exc:
            raise ImportError(
                "The 'vllm' package is required for the 'vllm' backend.  "
                "Install it with: pip install vllm"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt string.

        Args:
            prompt: The fully rendered prompt (system + context + question)
                produced by ``PromptBuilder``.

        Returns:
            The generated answer text (decoded, stripped of the prompt prefix).

        Raises:
            RuntimeError: When generation fails or times out.
        """
        if self._config.backend == "transformers":
            return self._generate_transformers(prompt)
        return self._generate_vllm(prompt)

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream generated tokens one-by-one for SSE support.

        Args:
            prompt: The fully rendered prompt string.

        Yields:
            Successive token strings as they are decoded.
        """
        if self._config.backend != "transformers":
            # Fallback: yield the full answer at once for non-streaming backends.
            yield self.generate(prompt)
            return

        if not _TRANSFORMERS_AVAILABLE:  # pragma: no cover
            yield self.generate(prompt)
            return

        inputs = self._tokenizer(prompt, return_tensors="pt")  # type: ignore[misc]
        inputs = {k: v.to(self._config.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self._tokenizer,  # type: ignore[arg-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        import threading

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self._config.max_new_tokens,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "repetition_penalty": self._config.repetition_penalty,
            "do_sample": self._config.temperature > 0,
        }
        thread = threading.Thread(
            target=self._model.generate,  # type: ignore[union-attr]
            kwargs=generation_kwargs,
        )
        thread.start()
        for token_text in streamer:
            yield token_text

    def _generate_transformers(self, prompt: str) -> str:
        """Run synchronous generation via the transformers backend."""
        inputs = self._tokenizer(prompt, return_tensors="pt")  # type: ignore[misc]
        inputs = {k: v.to(self._config.device) for k, v in inputs.items()}

        start = time.monotonic()
        with torch.no_grad():  # type: ignore[union-attr]
            output_ids = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                repetition_penalty=self._config.repetition_penalty,
                do_sample=self._config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
            )
        elapsed = time.monotonic() - start

        # Trim the prompt tokens from the output
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]
        decoded: str = self._tokenizer.decode(  # type: ignore[union-attr]
            generated_ids,
            skip_special_tokens=True,
        )
        logger.debug("Generation took %.2f s.", elapsed)
        return decoded.strip()

    def _generate_vllm(self, prompt: str) -> str:
        """Run generation via the vLLM engine."""
        from vllm import SamplingParams  # type: ignore[import]

        params = SamplingParams(
            max_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            repetition_penalty=self._config.repetition_penalty,
        )
        outputs = self._vllm_engine.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    @property
    def config(self) -> LLMConfig:
        """Return the active LLM configuration (read-only)."""
        return self._config


class MockLLMClient:
    """Deterministic stub for use in unit tests and CI without GPU.

    Accepts the same constructor signature as ``LLMClient`` and returns
    a hard-coded answer so that ``QAEngine``, ``PromptBuilder``, and
    ``CitationParser`` can be tested without loading model weights.
    """

    DEFAULT_RESPONSE = (
        "Theo Điều 10 Bộ luật Lao động 2019, người lao động có quyền làm việc "
        "và tự do lựa chọn việc làm. "
        "[Bộ luật Lao động 2019, Điều 10, Khoản 1]"
    )

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        fixed_response: Optional[str] = None,
    ) -> None:
        """Initialise the mock client.

        Args:
            config: Optional config object (stored but not used for generation).
            fixed_response: Custom response string.  Defaults to
                ``MockLLMClient.DEFAULT_RESPONSE``.
        """
        self._config = config or LLMConfig()
        self._response = fixed_response or self.DEFAULT_RESPONSE

    def generate(self, prompt: str) -> str:
        """Return the fixed response regardless of the input prompt.

        Args:
            prompt: Ignored in the mock implementation.

        Returns:
            The pre-configured fixed response string.
        """
        logger.debug("MockLLMClient.generate called (prompt length=%d).", len(prompt))
        return self._response

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Yield the fixed response as a single chunk.

        Args:
            prompt: Ignored in the mock implementation.

        Yields:
            The pre-configured fixed response string as a single token.
        """
        yield self._response

    @property
    def config(self) -> LLMConfig:
        """Return the active LLM configuration (read-only)."""
        return self._config
