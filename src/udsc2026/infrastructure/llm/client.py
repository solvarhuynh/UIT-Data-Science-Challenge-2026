"""Text generation client for configured checkpoints via Transformers or vLLM."""

import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar, cast

from udsc2026.infrastructure.llm.config import LLMConfig

logger = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")


def _run_with_timeout(
    operation: Callable[[], ResultT],
    timeout_seconds: float | None,
    *,
    on_complete: Callable[[], None] | None = None,
) -> ResultT:
    """Run a blocking call with a bounded wait and completion notification."""

    if timeout_seconds is None:
        try:
            return operation()
        finally:
            if on_complete is not None:
                on_complete()

    outcomes: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            outcome: tuple[bool, object] = (True, operation())
        except BaseException as exc:
            outcome = (False, exc)
        finally:
            if on_complete is not None:
                on_complete()
        outcomes.put(outcome)

    worker = threading.Thread(target=run, daemon=True, name="llm-generation")
    try:
        worker.start()
    except BaseException:
        if on_complete is not None:
            on_complete()
        raise
    try:
        succeeded, outcome = outcomes.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(
            f"LLM generation exceeded {timeout_seconds:g} seconds"
        ) from exc
    if succeeded:
        return cast(ResultT, outcome)
    if isinstance(outcome, BaseException):
        raise outcome
    raise RuntimeError("LLM generation returned an invalid worker outcome")


# ---------------------------------------------------------------------------
# Type stubs – transformers / torch are optional at import time so that unit
# tests (which mock this class) do not require GPU or heavy dependencies.
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:  # Transformers releases before multimodal auto classes.
        AutoModelForImageTextToText = None
    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False


class LLMClient:
    """Load and run a configured model for Vietnamese legal answer generation.

    Usage::

        config = LLMConfig(model_path="./models/qwen3.5-2b", device="cuda")
        client = LLMClient(config)
        answer = client.generate("Câu hỏi pháp luật của bạn...")

    The client serializes concurrent calls to ``generate`` when using the
    ``transformers`` backend.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialise the LLM client and load model weights from the configured checkpoint.

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
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._vllm_engine: Any | None = None
        self._generation_lock = threading.Lock()
        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load tokenizer and model weights from the configured checkpoint path."""
        if self._config.backend == "transformers":
            self._load_transformers()
        else:
            self._load_vllm()

    def _resolve_dtype(self) -> Any:
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
        quantization = self._config.quantization

        logger.info(
            "Loading tokenizer from '%s'…",
            model_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        load_kwargs: dict[str, Any] = {
            "device_map": "auto" if device == "cuda" else device,
            "trust_remote_code": True,
        }
        if quantization != "none":
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for model quantization "
                    "('4bit'/'8bit').  Install it with: pip install bitsandbytes"
                ) from exc
            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
        else:
            load_kwargs["torch_dtype"] = dtype

        logger.info(
            "Loading model from '%s' (dtype=%s, device=%s, quantization=%s)…",
            model_path,
            self._config.dtype,
            device,
            quantization,
        )
        model_config_path = Path(model_path) / "config.json"
        model_type = None
        if model_config_path.is_file():
            try:
                model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
                model_type = model_config.get("model_type")
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid model config file: {model_config_path}"
                ) from exc

        if model_type == "qwen3_5":
            if AutoModelForImageTextToText is None:
                raise RuntimeError(
                    "The configured Qwen3.5 checkpoint requires a Transformers "
                    "release that provides AutoModelForImageTextToText. Upgrade "
                    "the pinned Transformers dependency before using this model."
                )
            model_loader = AutoModelForImageTextToText
        else:
            model_loader = AutoModelForCausalLM

        model = model_loader.from_pretrained(
            model_path,
            **load_kwargs,
        )
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        logger.info("Model loaded successfully.")

    def _load_vllm(self) -> None:
        """Load model using the vLLM backend for high-throughput serving."""
        try:
            from vllm import LLM

            self._vllm_engine = LLM(
                model=self._config.model_path,
                dtype=self._config.dtype,
                trust_remote_code=True,
            )
            if _TRANSFORMERS_AVAILABLE:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._config.model_path,
                    trust_remote_code=True,
                )
            logger.info("vLLM engine initialised for '%s'.", self._config.model_path)
        except ImportError as exc:
            raise ImportError(
                "The 'vllm' package is required for the 'vllm' backend.  "
                "Install it with: pip install vllm"
            ) from exc

    def _require_model(self) -> Any:
        """Return the loaded transformers model or fail with a clear error."""
        if self._model is None:
            raise RuntimeError("The transformers model is not loaded")
        return self._model

    def _require_tokenizer(self) -> Any:
        """Return the loaded tokenizer or fail with a clear error."""
        if self._tokenizer is None:
            raise RuntimeError("The transformers tokenizer is not loaded")
        return self._tokenizer

    def _require_vllm_engine(self) -> Any:
        """Return the loaded vLLM engine or fail with a clear error."""
        if self._vllm_engine is None:
            raise RuntimeError("The vLLM engine is not loaded")
        return self._vllm_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _render_chat_messages(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> str:
        """Apply the checkpoint's native Qwen chat template."""

        if not messages:
            raise ValueError("messages must not be empty")
        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            role = message.get("role", "").strip()
            content = message.get("content", "").strip()
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"messages[{index}] has an unsupported role")
            if not content:
                raise ValueError(f"messages[{index}].content must not be empty")
            normalized.append({"role": role, "content": content})
        tokenizer = self._require_tokenizer()
        rendered = tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str) or not rendered.strip():
            raise RuntimeError("tokenizer returned an empty chat prompt")
        return rendered

    def generate_messages(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> str:
        """Generate from explicit system/user turns using the model template."""

        return self.generate(self._render_chat_messages(messages))

    def generate_messages_batch(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, str]]],
    ) -> list[str]:
        """Generate several chat completions in one model forward loop.

        The transformers backend uses padded inputs so decoder-only models can
        share the prefill and decode work across the batch.  The vLLM backend
        already performs continuous batching internally.
        """

        if not messages_batch:
            return []
        prompts = [self._render_chat_messages(messages) for messages in messages_batch]
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError(
                "LLM client is still processing an earlier generation request"
            )

        def operation() -> list[str]:
            if self._config.backend == "transformers":
                return self._generate_transformers_batch(prompts)
            return self._generate_vllm_batch(prompts)

        return _run_with_timeout(
            operation,
            self._config.timeout_seconds,
            on_complete=self._generation_lock.release,
        )

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
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError(
                "LLM client is still processing an earlier generation request"
            )

        def operation() -> str:
            if self._config.backend == "transformers":
                return self._generate_transformers(prompt)
            return self._generate_vllm(prompt)

        return _run_with_timeout(
            operation,
            self._config.timeout_seconds,
            on_complete=self._generation_lock.release,
        )

    def is_busy(self) -> bool:
        """Return True while a generation request is still in flight.

        A client that reports ``is_busy()`` for many consecutive calls after a
        timeout is likely wedged (the worker thread never finished), so callers
        such as evaluation scripts can abort early instead of looping over
        thousands of samples that fail immediately.
        """
        if self._generation_lock.acquire(blocking=False):
            self._generation_lock.release()
            return False
        return True

    def _deadline_stopping_criteria(self) -> dict[str, Any]:
        """Build a transformers deadline criterion when a timeout is configured."""

        timeout_seconds = self._config.timeout_seconds
        if timeout_seconds is None or not _TRANSFORMERS_AVAILABLE:
            return {}
        from transformers import StoppingCriteria, StoppingCriteriaList

        deadline = time.monotonic() + timeout_seconds

        class DeadlineStoppingCriteria(StoppingCriteria):
            def __call__(
                self,
                input_ids: Any,
                scores: Any,
                **kwargs: Any,
            ) -> bool:
                del input_ids, scores, kwargs
                return time.monotonic() >= deadline

        return {"stopping_criteria": StoppingCriteriaList([DeadlineStoppingCriteria()])}

    def _generate_transformers(self, prompt: str) -> str:
        """Run synchronous generation via the transformers backend."""
        tokenizer = self._require_tokenizer()
        model = self._require_model()
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._config.device) for k, v in inputs.items()}

        start = time.monotonic()

        def generate_ids() -> Any:
            with torch.no_grad():
                return model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_new_tokens,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    repetition_penalty=self._config.repetition_penalty,
                    do_sample=self._config.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    **self._deadline_stopping_criteria(),
                )

        output_ids = generate_ids()
        elapsed = time.monotonic() - start

        # Trim the prompt tokens from the output
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]
        decoded: str = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        logger.debug("Generation took %.2f s.", elapsed)
        return decoded.strip()

    def _generate_transformers_batch(self, prompts: Sequence[str]) -> list[str]:
        """Run one padded Transformers generation call for several prompts."""

        tokenizer = self._require_tokenizer()
        model = self._require_model()
        previous_padding_side = tokenizer.padding_side
        previous_pad_token = tokenizer.pad_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        try:
            inputs = tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._config.device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_new_tokens,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    repetition_penalty=self._config.repetition_penalty,
                    do_sample=self._config.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    **self._deadline_stopping_criteria(),
                )
            input_width = inputs["input_ids"].shape[1]
            return [
                tokenizer.decode(
                    output_ids[index][input_width:],
                    skip_special_tokens=True,
                ).strip()
                for index in range(len(prompts))
            ]
        finally:
            tokenizer.padding_side = previous_padding_side
            tokenizer.pad_token = previous_pad_token

    def _generate_vllm(self, prompt: str) -> str:
        """Run generation via the vLLM engine."""
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            repetition_penalty=self._config.repetition_penalty,
        )
        outputs = self._require_vllm_engine().generate([prompt], params)
        return cast(str, outputs[0].outputs[0].text).strip()

    def _generate_vllm_batch(self, prompts: Sequence[str]) -> list[str]:
        """Run a batch through vLLM's scheduler."""

        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            repetition_penalty=self._config.repetition_penalty,
        )
        outputs = self._require_vllm_engine().generate(list(prompts), params)
        return [cast(str, item.outputs[0].text).strip() for item in outputs]

    @property
    def config(self) -> LLMConfig:
        """Return the active LLM configuration (read-only)."""
        return self._config
