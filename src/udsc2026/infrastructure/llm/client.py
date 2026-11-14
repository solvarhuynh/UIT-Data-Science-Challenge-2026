"""Local LLM client wrapping the Qwen3 checkpoint via transformers or vLLM."""

import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generator, Optional, TypeVar, cast

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
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._vllm_engine: Any | None = None
        self._generation_lock = threading.Lock()
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
        model = AutoModelForCausalLM.from_pretrained(
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
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError(
                "LLM client is still processing an earlier generation request"
            )

        try:
            tokenizer = self._require_tokenizer()
            model = self._require_model()
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._config.device) for k, v in inputs.items()}

            poll_timeout = min(self._config.timeout_seconds or 0.5, 0.5)
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=poll_timeout,
            )
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": self._config.max_new_tokens,
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "repetition_penalty": self._config.repetition_penalty,
                "do_sample": self._config.temperature > 0,
            }
            generation_kwargs.update(self._deadline_stopping_criteria())
        except BaseException:
            self._generation_lock.release()
            raise
        generation_errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        generation_done = threading.Event()
        consumer_done = threading.Event()
        release_guard = threading.Lock()
        lock_released = False
        timeout_seconds = self._config.timeout_seconds
        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        )

        def release_when_finished() -> None:
            nonlocal lock_released
            if not generation_done.is_set() or not consumer_done.is_set():
                return
            with release_guard:
                if not lock_released:
                    lock_released = True
                    self._generation_lock.release()

        def generate_tokens() -> None:
            try:
                model.generate(**generation_kwargs)
            except BaseException as exc:
                generation_errors.put(exc)
                streamer.end()
            finally:
                generation_done.set()
                release_when_finished()

        thread = threading.Thread(
            target=generate_tokens,
            daemon=True,
            name="llm-stream-generation",
        )
        try:
            thread.start()
        except BaseException:
            self._generation_lock.release()
            raise
        try:
            iterator = iter(streamer)
            while True:
                try:
                    token_text = next(iterator)
                except StopIteration:
                    break
                except queue.Empty as exc:
                    if not generation_errors.empty():
                        raise RuntimeError("LLM streaming generation failed") from (
                            generation_errors.get_nowait()
                        )
                    if deadline is not None and time.monotonic() >= deadline:
                        if timeout_seconds is None:
                            raise RuntimeError(
                                "LLM streaming deadline exists without a timeout"
                            ) from exc
                        raise TimeoutError(
                            "LLM streaming generation exceeded "
                            f"{timeout_seconds:g} seconds"
                        ) from exc
                    continue
                yield cast(str, token_text)
        finally:
            consumer_done.set()
            release_when_finished()
        if not generation_errors.empty():
            raise RuntimeError("LLM streaming generation failed") from (
                generation_errors.get_nowait()
            )

    def generate_stream_messages(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> Generator[str, None, None]:
        """Stream from chat turns after applying the checkpoint template."""

        yield from self.generate_stream(self._render_chat_messages(messages))

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
