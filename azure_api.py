import logging
from typing import List, Dict, Tuple, Any
from retry import retry
import json
import openai
import base64
import imghdr
import httpx

import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format, augment_response_object
from anthropic import AnthropicFoundry

logger = logging.getLogger(__name__)

NAME = "azure"
GPT_KEY_NAME = "azure_gpt"
CLAUDE_KEY_NAME = "azure_claude"
MISTRAL_KEY_NAME = "azure_mistral"


class AzureBackend(backends.RemoteBackend):

    def __init__(self):
        key_registry = backends.KeyRegistry.from_json()

        self.key_name = NAME
        azure_key = key_registry.get_key_for(GPT_KEY_NAME)
        self.key = azure_key
        self.openai_client = self._build_openai_client(azure_key)

        claude_key = key_registry.get_key_for(CLAUDE_KEY_NAME)
        self.claude_client = self._build_claude_client(claude_key)

        mistral_key = key_registry.get_key_for(MISTRAL_KEY_NAME)
        self.mistral_client = self._build_mistral_client(mistral_key)

        # RemoteBackend expects self.client; point it at the OpenAI client
        self.client = self.openai_client

    @staticmethod
    def _build_openai_client(azure_key: dict) -> openai.AzureOpenAI:
        """Build an AzureOpenAI client (see run_azure_example.py).

        The key entry must provide ``api_version`` and either ``azure_endpoint``
        (the resource root, e.g. ``https://<resource>.cognitiveservices.azure.com/``)
        or a legacy ``base_url`` from which the endpoint is derived.
        """
        api_version = azure_key.get("api_version")
        if not api_version:
            raise ValueError(
                f"The '{GPT_KEY_NAME}' key entry is missing the required "
                f"'api_version' field (e.g. \"2024-12-01-preview\").")

        azure_endpoint = azure_key.get("azure_endpoint")
        if not azure_endpoint:
            base_url = azure_key.get("base_url")
            if not base_url:
                raise ValueError(
                    f"The '{GPT_KEY_NAME}' key entry must provide either "
                    f"'azure_endpoint' or 'base_url'.")
            # Derive the resource root from a legacy base_url such as
            # https://<resource>.openai.azure.com/openai/v1/
            azure_endpoint = base_url.split("/openai/")[0].rstrip("/") + "/"

        return openai.AzureOpenAI(
            api_key=azure_key["api_key"],
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )

    @staticmethod
    def _build_claude_client(claude_key: dict) -> AnthropicFoundry:
        """Build an AnthropicFoundry client for Azure-hosted Claude models.

        See run_azure_claude_example.py: the client only needs ``api_key`` and
        the Foundry ``base_url`` (e.g.
        ``https://<resource>.services.ai.azure.com/anthropic/``); the SDK manages
        the API version itself.
        """
        return AnthropicFoundry(
            api_key=claude_key["api_key"],
            base_url=claude_key["base_url"],
        )

    @staticmethod
    def _build_mistral_client(mistral_key: dict) -> openai.OpenAI:
        """Build a plain OpenAI client for Azure-hosted Mistral models.

        See run_azure_mistral_example.py: Mistral deployments are served via the
        OpenAI-compatible v1 endpoint
        (e.g. ``https://<resource>.openai.azure.com/openai/v1/``), so a plain
        ``openai.OpenAI`` client with ``base_url`` is used instead of
        ``AzureOpenAI`` (no ``api_version`` needed).
        """
        return openai.OpenAI(
            api_key=mistral_key["api_key"],
            base_url=mistral_key["base_url"],
        )

    def _make_api_client(self):
        # Clients are initialised directly in __init__; this satisfies the abstract method.
        pass

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        if "claude" in model_spec.model_id.lower():
            return AzureClaudeModel(self.claude_client, model_spec)
        if "mistral" in model_spec.model_id.lower():
            return AzureOpenAIModel(self.mistral_client, model_spec)
        return AzureOpenAIModel(self.openai_client, model_spec)


# ---------------------------------------------------------------------------
# Shared image-encoding helper (identical logic for both model families)
# ---------------------------------------------------------------------------

def encode_image(image_path):
    if image_path.startswith('http'):
        image_bytes = httpx.get(image_path).content
        image_type = imghdr.what(None, image_bytes)
        return True, image_path, image_type
    with open(image_path, "rb") as image_file:
        image_type = imghdr.what(image_path)
        return False, base64.b64encode(image_file.read()).decode('utf-8'), 'image/' + str(image_type)


# ---------------------------------------------------------------------------
# GPT / OpenAI model
# ---------------------------------------------------------------------------

class AzureOpenAIModel(backends.Model):

    def __init__(self, client: openai.OpenAI, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.client = client

    def encode_messages(self, messages) -> list:
        encoded_messages = []
        for message in messages:
            if "image" not in message.keys():
                encoded_messages.append(message)
            else:
                this = {"role": message["role"],
                        "content": [{"type": "text",
                                     "text": message["content"].replace(" <image> ", " ")}]}

                if 'multimodality' not in self.model_spec.model_config:
                    logger.info(
                        f"The backend {self.model_spec.model_id} does not support multimodal inputs!")
                    raise Exception(
                        f"The backend {self.model_spec.model_id} does not support multimodal inputs!")

                if not self.model_spec['model_config']['multimodality']['multiple_images'] \
                        and len(message['image']) > 1:
                    logger.info(f"The backend {self.model_spec.model_id} does not support multiple images!")
                    raise Exception(f"The backend {self.model_spec.model_id} does not support multiple images!")

                for image in message['image']:
                    is_url, loaded, image_type = encode_image(image)
                    if is_url:
                        this["content"].append(dict(type="image_url", image_url={"url": loaded}))
                    else:
                        this["content"].append(dict(type="image_url",
                                                    image_url={"url": f"data:{image_type};base64,{loaded}"}))
                encoded_messages.append(this)
        return encoded_messages

    @retry(tries=3, delay=90, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        prompt = self.encode_messages(messages)
        gen_kwargs = dict(model=self.model_spec.model_id,
                          messages=prompt,
                          temperature=self.temperature,
                          max_completion_tokens=self.max_tokens)
        if "mistral" in self.model_spec.model_id.lower():
            # Non-OpenAI models on the OpenAI-compatible v1 endpoint use the
            # classic `max_tokens` parameter instead of `max_completion_tokens`.
            del gen_kwargs["max_completion_tokens"]
            gen_kwargs["max_tokens"] = self.max_tokens
        model_config = getattr(self.model_spec, "model_config", {})
        if 'reasoning_model' in model_config:
            if 'reasoning_effort' in model_config:
                gen_kwargs['reasoning_effort'] = model_config['reasoning_effort']
            if not self.temperature > 0:
                raise ValueError(
                    f"For reasoning models temperature must be >0, but is {self.temperature}. "
                    f"Please use the -t option to set a temperature and try again.")
            del gen_kwargs["max_completion_tokens"]

        if self.model_spec["model_name"] == "upgpt-codex":
            api_response = self.client.responses.create(
                model=self.model_spec["model_id"], input=prompt)
            response_text = api_response.output_text.strip()
            response = json.loads(api_response.json())
        else:
            api_response = self.client.chat.completions.create(**gen_kwargs)
            message = api_response.choices[0].message
            if message.role != "assistant":
                raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
            response_text = message.content.strip()
            response = json.loads(api_response.json())

        # Token-fertility logging: the Azure-hosted GPT model has no public
        # tokenizer, so the API-reported token counts are the only exact source.
        # completion_tokens is the raw total (visible text + reasoning); subtract
        # the reasoning tokens to obtain the visible-output token count.
        usage = getattr(api_response, "usage", None)
        if usage is not None:
            output_tokens = getattr(usage, "completion_tokens", None)
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
            text_tokens = (output_tokens - reasoning_tokens
                           if output_tokens is not None else None)
            logger.info(
                "[fertility] model=%s output_tokens=%s reasoning_tokens=%s "
                "text_tokens=%s input_tokens=%s",
                self.model_spec.model_id, output_tokens, reasoning_tokens,
                text_tokens, getattr(usage, "prompt_tokens", None),
            )

        return prompt, response, response_text


# ---------------------------------------------------------------------------
# Claude / Anthropic model
# ---------------------------------------------------------------------------

class AzureClaudeModel(backends.Model):

    def __init__(self, client: AnthropicFoundry, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.client = client

    def encode_messages(self, messages) -> list:
        encoded_messages = []
        for message in messages:
            if "image" not in message.keys():
                encoded_messages.append({"role": message["role"], "content": message["content"]})
            else:
                this = {"role": message["role"],
                        "content": [{"type": "text",
                                     "text": message["content"].replace(" <image> ", " ")}]}

                if 'multimodality' not in self.model_spec.model_config:
                    logger.info(
                        f"The backend {self.model_spec.model_id} does not support multimodal inputs!")
                    raise Exception(
                        f"The backend {self.model_spec.model_id} does not support multimodal inputs!")

                if not self.model_spec['model_config']['multimodality']['multiple_images'] \
                        and len(message['image']) > 1:
                    logger.info(f"The backend {self.model_spec.model_id} does not support multiple images!")
                    raise Exception(f"The backend {self.model_spec.model_id} does not support multiple images!")

                for image in message['image']:
                    is_url, loaded, image_type = encode_image(image)
                    if is_url:
                        this["content"].append(dict(type="image_url", image_url={"url": loaded}))
                    else:
                        this["content"].append(dict(type="image_url",
                                                    image_url={"url": f"data:{image_type};base64,{loaded}"}))
                encoded_messages.append(this)
        return encoded_messages

    @retry(tries=3, delay=90, logger=logger)
    @augment_response_object
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        prompt = self.encode_messages(messages)
        gen_kwargs = dict(model=self.model_spec.model_id,
                          messages=prompt,
                          temperature=self.temperature,
                          max_tokens=self.max_tokens)
        model_config = getattr(self.model_spec, "model_config", {})
        if 'thinking_mode' in model_config:
            # The Azure AI Foundry Claude endpoint does NOT support the standard
            # `thinking: {type: "enabled", budget_tokens: N}` API. It returns:
            #   "thinking.type.enabled" is not supported for this model.
            #   Use "thinking.type.adaptive" and "output_config.effort".
            # So thinking is controlled via an effort level (low|medium|high),
            # the same control surface as the GPT reasoning models, rather than
            # an explicit token budget.
            # Thinking-mode constraint: temperature must be 1.
            gen_kwargs["temperature"] = 1.
            effort = model_config.get('thinking_effort', 'medium')
            gen_kwargs["extra_body"] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            }

        api_response = self.client.messages.create(**gen_kwargs)
        # With thinking enabled the response contains a thinking block followed
        # by the text block, so select the first text block rather than relying
        # on a fixed index.
        response_text = next(
            (block.text for block in api_response.content
             if getattr(block, "type", None) == "text"),
            api_response.content[-1].text,
        )

        # Token-fertility logging: the Azure-hosted Claude model has no public
        # tokenizer, so the API-reported token counts are the only exact source.
        # output_tokens is the raw total (visible text + thinking); subtract the
        # thinking tokens to obtain the visible-output token count.
        usage = getattr(api_response, "usage", None)
        if usage is not None:
            output_tokens = getattr(usage, "output_tokens", None)
            details = getattr(usage, "output_tokens_details", None)
            thinking_tokens = getattr(details, "thinking_tokens", 0) or 0
            text_tokens = (output_tokens - thinking_tokens
                           if output_tokens is not None else None)
            logger.info(
                "[fertility] model=%s output_tokens=%s thinking_tokens=%s "
                "text_tokens=%s input_tokens=%s",
                self.model_spec.model_id, output_tokens, thinking_tokens,
                text_tokens, getattr(usage, "input_tokens", None),
            )

        response = api_response.model_dump(mode="json")
        return prompt, response, response_text
