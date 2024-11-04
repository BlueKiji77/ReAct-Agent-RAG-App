
from typing import List
import asyncio
from contextlib import contextmanager
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Sequence,
    cast,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload

# dispatcher setup
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.llm import (
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMChatInProgressEvent,
)

from llama_index.core.instrumentation.events.base import BaseEvent

class LLMCompletionListEndEvent(BaseEvent):
    prompt: str
    response: List[CompletionResponse]

    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMCompletionListEndEvent"

dispatcher = get_dispatcher(__name__)

def llm_completion_callback() -> Callable:
    def wrap(f: Callable) -> Callable:
        @contextmanager
        def wrapper_logic(_self: Any) -> Generator[CallbackManager, None, None]:
            callback_manager = getattr(_self, "callback_manager", None)
            if not isinstance(callback_manager, CallbackManager):
                raise ValueError(
                    "Cannot use llm_completion_callback on an instance "
                    "without a callback_manager attribute."
                )

            yield callback_manager

        async def wrapped_async_llm_predict(
            _self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager:
                span_id = dispatcher.root.current_span_ids or ""
                dispatcher.event(
                    LLMCompletionStartEvent(
                        model_dict=_self.to_dict(),
                        prompt=str(args[0]),
                        additional_kwargs=kwargs,
                        span_id=span_id,
                    )
                )
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.PROMPT: args[0],
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )

                f_return_val = await f(_self, *args, **kwargs)

                if isinstance(f_return_val, AsyncGenerator):
                    # intercept the generator and add a callback to the end
                    async def wrapped_gen() -> CompletionResponseAsyncGen:
                        last_response = None
                        async for x in f_return_val:
                            dispatcher.event(
                                LLMCompletionEndEvent(
                                    prompt=str(args[0]),
                                    response=x,
                                    span_id=span_id,
                                )
                            )
                            yield cast(CompletionResponse, x)
                            last_response = x

                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.PROMPT: args[0],
                                EventPayload.COMPLETION: last_response,
                            },
                            event_id=event_id,
                        )

                    return wrapped_gen()
                else:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            EventPayload.PROMPT: args[0],
                            EventPayload.COMPLETION: f_return_val,
                        },
                        event_id=event_id,
                    )
                    dispatcher.event(
                        LLMCompletionEndEvent(
                            prompt=str(args[0]),
                            response=f_return_val,
                            span_id=span_id,
                        )
                    )

            return f_return_val

        def wrapped_llm_predict(_self: Any, *args: Any, **kwargs: Any) -> Any:
            with wrapper_logic(_self) as callback_manager:
                span_id = dispatcher.root.current_span_ids or ""
                dispatcher.event(
                    LLMCompletionStartEvent(
                        model_dict=_self.to_dict(),
                        prompt=str(args[0]),
                        additional_kwargs=kwargs,
                        span_id=span_id,
                    )
                )
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.PROMPT: args[0],
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )

                f_return_val = f(_self, *args, **kwargs)
                if isinstance(f_return_val, Generator):
                    # intercept the generator and add a callback to the end
                    def wrapped_gen() -> CompletionResponseGen:
                        last_response = None
                        for x in f_return_val:
                            dispatcher.event(
                                LLMCompletionEndEvent(
                                    prompt=str(args[0]), response=x, span_id=span_id
                                )
                            )
                            yield cast(CompletionResponse, x)
                            last_response = x

                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.PROMPT: args[0],
                                EventPayload.COMPLETION: last_response,
                            },
                            event_id=event_id,
                        )

                    return wrapped_gen()
                    
                if isinstance(f_return_val, list):
                    responses = f_return_val
                else:
                    responses = [f_return_val]                                                           
                callback_manager.on_event_end(
                    CBEventType.LLM,
                    payload={
                        EventPayload.PROMPT: args[0],
                        EventPayload.COMPLETION: [response for response in responses],
                     },
                        event_id=event_id,
                )
                dispatcher.event(
                    LLMCompletionListEndEvent(
                        prompt=str(args[0]),
                        response=[response for response in responses],
                        span_id=span_id,
                    )
                )
            return [response.dict() for response in responses]

        async def async_dummy_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            return await f(_self, *args, **kwargs)

        def dummy_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            return f(_self, *args, **kwargs)

        # check if already wrapped
        is_wrapped = getattr(f, "__wrapped__", False)
        if not is_wrapped:
            f.__wrapped__ = True  # type: ignore

        if asyncio.iscoroutinefunction(f):
            if is_wrapped:
                return async_dummy_wrapper
            else:
                return wrapped_async_llm_predict
        else:
            if is_wrapped:
                return dummy_wrapper
            else:
                return wrapped_llm_predict

    return wrap

### TO model.py FILE

import torch
import os
from tqdm import tqdm
from threading import Thread
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.base.llms.types import (
    MessageRole,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    LLMMetadata
        )

# from llama_index.core.base.llms.generic_utils import completion_response_to_chat_response
from llama_index.core.llms.callbacks import llm_chat_callback, CallbackManager #, llm_completion_callback
from llama_index.core.utils import get_cache_dir
from typing import Literal, Optional, Union, Dict, Any, List, Sequence, Callable
from llama_index.core.llms.custom import CustomLLM
from dsp.modules.lm import LM
from llama_cpp import Llama
import requests
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate #, PromptTemplate
from llama_index.core.instrumentation.events.llm import (
    LLMPredictEndEvent,
    LLMPredictStartEvent,
    )

logger = logging.getLogger(__name__)

DEFAULT_HUGGINGFACE_MODEL = 'llama_3.1_8b'

###############################################################################
def completion_response_to_chat_response(
    completion_response: CompletionResponse,
) -> ChatResponse:
    """Convert a completion response to a chat response."""
    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=completion_response["text"],
            additional_kwargs=completion_response["additional_kwargs"],
        ),
        raw=completion_response["raw"],)

################################################################################

class LLM_WRAPPER(LLM):
#     model_config = ConfigDict(protected_namespaces=())
    model_config = {"protected_namespaces": ()}
    model_config['protected_namespaces'] = ()
    
    model_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The model name to use from HuggingFace. "
            "Unused if `model` is passed in directly."
        ),
    )
    model_url: Optional[str] = Field(default=None,
                                    description="The URL llama-cpp model to download and use."
    )
    model_path: Optional[str] = Field(default=None,
                                    description="The path to the llama-cpp model to use."
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )

    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for LLM calls."
    )
    messages_to_prompt: Callable = Field(
        description="Function to convert a list of messages to an LLM prompt.",
        default=None,
        exclude=True,
    )
    completion_to_prompt: Callable = Field(
        description="Function to convert a completion to an LLM prompt.",
        default=None,
        exclude=True,
    )
    verbose: bool = Field(
        default=False,
        description="Whether to print verbose output.",
    )
    device_map: str = Field(
        default="auto", description="The device_map to use. Defaults to 'auto'."
    )
    stopping_ids: List[int] = Field(
        default_factory=list,
        description=(
            "The stopping ids to use. "
            "Generation stops when these token IDs are predicted."
        ),
    )
    tokenizer_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The name of the tokenizer to use from HuggingFace. "
            "Unused if `tokenizer` is passed in directly."
        ),
    )
    tokenizer_outputs_to_remove: list = Field(
        default_factory=list,
        description=(
            "The outputs to remove from the tokenizer. "
            "Sometimes huggingface tokenizers return extra inputs that cause errors."
        ),
    )
    tokenizer_kwargs: dict = Field(
        default_factory=dict, description="The kwargs to pass to the tokenizer."
    )
    special_tokens: dict = Field(
        default_factory=dict, description="Special tokens, custom for prompt formatting."
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=(
            "Wether the model has function calling capability" + " Be sure to verify that you either pass an appropriate tokenizer that is suitable for function calling."
        ),
    )
    provider: str = Field(
        default="local",
        description=("LLM provider"
        ),
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of LLM calls"
    ),
    device: Optional[Union[str, torch.device]] = Field(
        default=None,
        description="Compute device"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="DSPy args"
    )
    _model: Any = PrivateAttr()
    _context_window: int = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _stopping_criteria: Any = PrivateAttr()
    _gguf: bool = PrivateAttr()

    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        model_url: Optional[str] = None,
        model_path: Optional[str] = None,
        model_kwargs: Optional[dict] = {},
        generate_kwargs: Optional[dict] = {},
        tokenizer_kwargs: Optional[dict] = {},
        verbose: Optional[bool] = False,
        is_function_calling_model: bool=False,
        provider: Optional[str] = "local",

        # Prompt params
        messages_to_prompt = None, 
        completion_to_prompt =  None, 
        system_prompt: Optional[str] = None,
        special_tokens: Optional[dict] = None,

        stopping_ids: Optional[List[int]] = None,
        callback_manager: Optional[CallbackManager] = None,
        token: Optional[str] = None,
        hf_device_map: Literal[
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ] = None,
    ):
        """Wrapper for HuggingFace and LlamaCPP-python models, interface to DSPy"""

        super().__init__(
        )

        # Variables setting
        self.model_url = model_url
        self.tokenizer_name = tokenizer_name
        self.history = []
        self.generate_kwargs = {**(generate_kwargs or {})}
        self.device_map = model_kwargs.pop("device_map") if  model_kwargs.get("device_map") else hf_device_map
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.special_tokens = special_tokens
        self.system_prompt = system_prompt
        self.is_function_calling_model = is_function_calling_model
        self.provider = provider

        # Check if both model_name and model_url are None
        if model_name is None and model_url is None:
            raise ValueError("Provide a value for either model_name or model_url.")
        if model_name:
            model_name = model_name
        elif model_url:
            model_name = os.path.basename(model_url)

        self.model_name = model_name
        self.kwargs = {
            "model": model_name,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }

        context_window = tokenizer_kwargs["max_length"]
        model_kwargs = dict(token=token or os.environ.get("HF_TOKEN"), **model_kwargs)

        if model_url: # Checks if model is to be loaded in gguf formmat
            self._gguf = True
            
            model_url = model_url or self._get_model_path_for_version()

            # check if model is cached
            if model_path is not None:
                if not os.path.exists(model_path):
                    raise ValueError(
                        "Provided model path does not exist. "
                        "Please check the path or provide a model_url to download."
                    )
                else:
                    
                    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                    self._model = Llama(model_path=model_path,
                                        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=3),
                                        **model_kwargs)
            else:
                cache_dir = get_cache_dir()
                model_path = os.path.join(cache_dir, "models", model_name)
                if not os.path.exists(model_path):
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    self._download_url(model_url, model_path)
                    assert os.path.exists(model_path)
                    
                from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                self._model = Llama(model_path=model_path,
                                        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=3),
                                        **model_kwargs)
            model_path = model_path
        elif self.device_map: # load gptq or full model
            self._gguf = False
            self._model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device_map, **model_kwargs)
        else:
            self._gguf = False
            self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(self.device)

        # check context_window
        if model_url:
            model_config_context = self._model._n_ctx
            model_context_window = model_config_context or context_window
        else:
            config_dict = self._model.config.to_dict()
            model_context_window = int(config_dict.get("max_position_embeddings", context_window))

        if model_context_window and model_context_window < context_window:
            logger.warning(
                f"Supplied context_window {context_window} is greater "
                f"than the model's max input size {model_context_window}. "
                "Disable this warning by setting a lower context_window."
            )
            context_window = model_context_window

        self._context_window = model_context_window

        if "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = context_window

        if tokenizer_name != model_name:
            logger.warning(
                f"The model `{model_name}` and tokenizer `{tokenizer_name}` "
                f"are different, please ensure that they are compatible.")
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, **tokenizer_kwargs)


        def _messages_to_prompt(messages,
                                add_gen_prompt=True,
                               ):
            prompt_templated = self._tokenizer.apply_chat_template(
                                                             messages,
                                                             tokenize=False,
                                                             add_generation_prompt=add_gen_prompt)
            return prompt_templated


        def _completion_to_prompt(
                               prompt,
                               add_gen_prompt=True,
                               special_tokens: Optional[dict] = {},
                               system_prompt: Optional[str] = "") -> str:
            messages = [dict(role = "user", content = prompt)]
            prompt_templated = self._tokenizer.apply_chat_template(
                                                             messages,
                                                             tokenize=False,
                                                             add_generation_prompt=add_gen_prompt)
            return prompt_templated

        self.messages_to_prompt = _messages_to_prompt or messages_to_prompt
        self.completion_to_prompt = _completion_to_prompt or completion_to_prompt

        # setup stopping criteria
        stopping_ids_list = stopping_ids or []


        class StopOnTokens(StoppingCriteria):
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs: Any,
            ) -> bool:
                for stop_id in stopping_ids_list:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        self._stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
                context_window=self._context_window,
                num_output=self.generate_kwargs.get("max_new_tokens", 128),  # Use get() with a default value
                model_name=self.model_name,  # Use self.model_name
                is_function_calling_model=self.is_function_calling_model
            )
    
    #####################################
    ########## General Methods ##########
    #####################################
    @classmethod
    def class_name(cls) -> str:
            return "DSPy_LlamaIndex_LLM"

    def _get_model_path_for_version(self) -> str:
        """Get model path for the current llama-cpp version."""
        import pkg_resources

        version = pkg_resources.get_distribution("llama-cpp-python").version
        major, minor, patch = version.split(".")

        # NOTE: llama-cpp-python<=0.1.78 supports GGML, newer support GGUF
        if int(major) <= 0 and int(minor) <= 1 and int(patch) <= 78:
            return DEFAULT_LLAMA_CPP_GGML_MODEL
        else:
            return DEFAULT_LLAMA_CPP_GGUF_MODEL


    def _download_url(self, model_url: str, model_path: str) -> None:
        completed = False
        try:
            print("Downloading url", model_url, "to path", model_path)
            with requests.get(model_url, stream=True) as r:
                with open(model_path, "wb") as file:
                    total_size = int(r.headers.get("Content-Length") or "0")
                    if total_size < 1000 * 1000:
                        raise ValueError(
                                    "Content should be at least 1 MB, but is only",
                                    r.headers.get("Content-Length"),
                                    "bytes",
                                        )
                    print("total size (MB):", round(total_size / 1000 / 1000, 2))
                    chunk_size = 1024 * 1024  # 1 MB
                    for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=int(total_size / chunk_size),):
                        file.write(chunk)
            completed = True
        except Exception as e:
            print("Error downloading model:", e)
        finally:
            if not completed:
                print("Download incomplete.", "Removing partially downloaded file.")
                os.remove(model_path)
                raise ValueError("Download incomplete.")


    def _args_to_hf(self, **kwargs):
        hf_kwargs = {}
        for k, v in kwargs.items():
            if k == "n":
                hf_kwargs["num_return_sequences"] = v
            elif k == "max_new_tokens":
                hf_kwargs["max_new_tokens"] = v
            elif k == "frequency_penalty":
                hf_kwargs["repetition_penalty"] = 1.0 - v
            elif k == "presence_penalty":
                hf_kwargs["diversity_penalty"] = v
            elif k == "max_tokens":
                hf_kwargs["max_new_tokens"] = v
            elif k == "temperature":
                hf_kwargs["temperature"] = v
            elif k in ["model", "add_gen_prompt"]:
                continue
            else:
                hf_kwargs[k] = v
        return hf_kwargs


    def _args_to_llamacpp(self, **kwargs):
        llamacpp_args = {}
        for key, value in kwargs.items():
            if key in ["model", "do_sample"]:
                continue
            elif key in ["max_new_tokens", "max_tokens"]:
                llamacpp_args["max_tokens"] = value
            elif key == "n":
                llamacpp_args["num_return_sequences"] = value
            else:
                llamacpp_args[key] = value
        return llamacpp_args


    ##################################
    ########## DSPy Methods ##########
    ##################################
    def print_green(self, text: str, end: str = "\n"):
        return "\x1b[32m" + str(text) + "\x1b[0m" + end

    def print_red(self, text: str, end: str = "\n"):
        return "\x1b[31m" + str(text) + "\x1b[0m" + end

    def inspect_history(self, n: int = 1, skip: int = 0):
        """Prints the last n prompts and their completions.

        TODO: print the valid choice that contains filled output field instead of the first.
        """
        provider: str = self.provider

        last_prompt = None
        printed = []
        n = n + skip

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]

            if prompt != last_prompt:
                if provider == "clarifai" or provider == "google" or provider == "groq" or provider == "Bedrock" or provider == "Sagemaker" or provider == "local":
                    printed.append((prompt, x["response"]))
                elif provider == "anthropic":
                    blocks = [{"text": block.text} for block in x["response"].content if block.type == "text"]
                    printed.append((prompt, blocks))
                elif provider == "cohere":
                    printed.append((prompt, x["response"].text))
                elif provider == "mistral":
                    printed.append((prompt, x['response'].choices))
                else:
                    printed.append((prompt, x["response"]["choices"]))

            last_prompt = prompt

            if len(printed) >= n:
                break

        printing_value = ""
        for idx, (prompt, choices) in enumerate(reversed(printed)):
            # skip the first `skip` prompts
            if (n - idx - 1) < skip:
                continue
            printing_value += "\n\n\n"
            printing_value += prompt

            text = ""
            if provider == "cohere" or provider == "Bedrock" or provider == "Sagemaker":
                text = choices
            elif provider == "openai" or provider == "ollama":
                text = ' ' + self._get_choice_text(choices[0]).strip()
            elif provider == "clarifai" or provider == "claude" :
                text=choices
            elif provider == "groq":
                text = ' ' + choices
            elif provider == "google":
                text = choices[0].parts[0].text
            elif provider == "mistral":
                text = choices[0].message.content
            else:
                text = choices[0]["text"]
            printing_value += self.print_green(text, end="")

            if len(choices) > 1:
                printing_value += self.print_red(f" \t (and {len(choices)-1} other completions)", end="")

            printing_value += "\n\n\n"

        return printing_value

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop("model")

        return self.__class__(model=model, **kwargs)

    def _generate(self, prompt, **kwargs):
        return self.complete(prompt, **kwargs)


    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)


    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.generate_kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
              }
        self.history.append(history)

        return response


    ########################################
    ########## LlamaIndex Methods ##########
    ########################################
    @dispatcher.span
    def predict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> str:
        """Predict for a given prompt.

        Args:
            prompt (BasePromptTemplate):
                The prompt to use for prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            str: The prediction output.

        Examples:
            ```python
            from llama_index.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please write a random name related to {topic}.")
            output = llm.predict(prompt, topic="cats")
            print(output)
            ```
        """
        dispatch_event = dispatcher.get_dispatch_event()

        dispatch_event(LLMPredictStartEvent(template=prompt))
        self._log_template_data(prompt, **prompt_args)

        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = self.chat(messages)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = self.complete(formatted_prompt, formatted=True)
            output = response[0]["text"]
        dispatch_event(LLMPredictEndEvent(output=output))
        return self._parse_output(output)


    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool=False, **kwargs: Any
    ) -> List[CompletionResponse]:
        """Completion endpoint."""
        full_prompt = prompt

        kwargs = {**self.generate_kwargs, **kwargs}

        add_gen_prompt = kwargs.pop("add_gen_prompt")
        if not self._gguf:
            if not formatted:
                full_prompt = self.completion_to_prompt(full_prompt,
                                                     add_gen_prompt=add_gen_prompt)
            inputs = self._tokenizer(full_prompt, return_tensors="pt")
            inputs = inputs.to(self._model.device)

            # remove keys from the tokenizer if needed, to avoid HF errors
            for key in self.tokenizer_outputs_to_remove:
                if key in inputs:
                    inputs.pop(key, None)

            generate_kwargs = self._args_to_hf(**kwargs)
            superclass_kwargs = self._args_to_hf(**self.kwargs)
            generation_kwargs = {key: value for key, value in superclass_kwargs.items() if key not in generate_kwargs.keys()}
            generation_kwargs = {**generation_kwargs, **generate_kwargs}
            temperature_arg = generation_kwargs.get('temperature')
            if temperature_arg == 0.0:
                generation_kwargs['do_sample'] = False
            
            outputs = self._model.generate(
                **inputs,
                stopping_criteria=self._stopping_criteria,
                **generation_kwargs,
                        )

            # Assuming inputs is a dictionary with 'input_ids' key
            if 'input_ids' in inputs.keys():
                input_length = inputs['input_ids'].shape[1]  # Assuming the shape is (batch_size, seq_length)
            else:
                # If inputs is not a dictionary, assume it's a list or tensor
                input_length = len(inputs)

            # Ensure outputs is a tensor
            outputs = torch.tensor(outputs) if not torch.is_tensor(outputs) else outputs

            # Slice the outputs tensor along the second dimension starting from input_length
            outputs = outputs[:, input_length:]

            completions = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

            return [CompletionResponse(text=completion) for completion in completions]


        # Interface to LlamaCPP
        elif self._gguf:
            self.generate_kwargs.update({"stream": False})
            generated_tokens = []
            generate_kwargs = self._args_to_llamacpp(**kwargs)
            superclass_kwargs = self._args_to_llamacpp(**self.kwargs)
            generation_kwargs = {key: value for key, value in superclass_kwargs.items() if key not in generate_kwargs.keys()}
            generation_kwargs = {**generation_kwargs, **generate_kwargs}
            num_return_sequences = generation_kwargs.pop("num_return_sequences")

            completions = []
            temperature_arg = generation_kwargs.get('temperature')

            if temperature_arg == 0.0:
                print(f"Temperature is {temperature_arg}. \nSetting 'do_sample = True' ")

            for i in range(num_return_sequences):
                completion = self._model(full_prompt, **generation_kwargs)
                completions.append(completion)
                
            test_responses = [CompletionResponse(text=completion["choices"][0]['text'], raw=completion)
                           for completion in completions]

            return test_responses

    @llm_chat_callback()
    def chat(self, messages: Sequence[Union[ChatMessage, Dict]], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=False, **kwargs)[0]
        return completion_response_to_chat_response(completion_response)


    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response_gen = self.stream_complete(prompt, formatted=False, **kwargs)
        return stream_completion_response_to_chat_response(completion_response_gen)


    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)


    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for message in self.stream_chat(messages, **kwargs):
                yield message
        # NOTE: convert generator to async generator
        return gen()


    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, to_foformattedrmat: bool = True, **kwargs: Any
        ) -> CompletionResponse:

        full_prompt = prompt

        if formatted:
            full_prompt = self._completion_to_prompt(full_prompt)

        if self._gguf:
            self.generate_kwargs.update({"stream": True})

            generate_kwargs = self._args_to_llamacpp(**kwargs)
            superclass_kwargs = self._args_to_llamacpp(**self.kwargs)
            generation_kwargs = {key: value for key, value in superclass_kwargs.items() if key not in generate_kwargs.keys()}
            generation_kwargs = {**generation_kwargs, **generate_kwargs}
            num_sequences_to_return = generation_kwargs.pop("n")
            tokens_to_generate = generation_kwargs.pop("max_tokens", 512)

            completion_iterators = []

            for i in range(num_sequences_to_return):
                completion_iter = self._model(full_prompt **generation_kwargs)
                completion_iterators.append(completion_iter)

            def gen() -> CompletionResponseGen:
                text = ""
                for completion_iterator in completion_iterators:
                    for completion in completion_iterator:
                        delta = completion["choices"][0]["text"]
                        text += delta
                        yield CompletionResponse(delta=delta, text=text, raw=response)

            return gen

        else:
            from transformers import TextIteratorStreamer
            inputs = self._tokenizer(full_prompt, return_tensors="pt")
            inputs = inputs.to(self._model.device)

            # remove keys from the tokenizer if needed, to avoid HF errors
            for key in self.tokenizer_outputs_to_remove:
                if key in inputs:
                    inputs.pop(key, None)

            generate_kwargs = self._args_to_hf(**kwargs)
            superclass_kwargs = self._args_to_hf(**self.kwargs)
            generation_kwargs = {key: value for key, value in superclass_kwargs.items() if key not in generate_kwargs.keys()}
            generation_kwargs = {**generation_kwargs, **generate_kwargs}

            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            generation_kwargs = dict(
                                    inputs,
                                    streamer=streamer,
                                    stopping_criteria=self._stopping_criteria,
                                    **generation_kwargs
                                        )

            # generate in background thread
            # NOTE/TODO: token counting doesn't work with streaming
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()

            # create generator based off of streamer
            def gen() -> CompletionResponseGen:
                text = ""
                for x in streamer:
                    text += x
                    yield CompletionResponse(text=text, delta=x)

            return gen()

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = True, **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, formatted=formatted, **kwargs)


    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = True, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            for message in self.stream_complete(prompt, formatted=formatted, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()


    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
            assert only_completed, "for now"
            assert return_sorted is False, "for now"

            if kwargs.get("n", 1) > 1 or kwargs.get("temperature", 0.0) > 0.1:
                kwargs["do_sample"] = True

            response = self.request(prompt, **kwargs)
            return [answer["text"] for answer in response]
