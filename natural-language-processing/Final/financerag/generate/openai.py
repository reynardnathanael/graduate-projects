import pdb
import logging
import multiprocessing
import os
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple, cast

import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from financerag.common.protocols import Generator

# We don't need the real key, but we set a dummy one just in case
openai.api_key = "sk-dummy"

logger = logging.getLogger(__name__)


class OpenAIGenerator(Generator):
    """
    A class that interfaces with a LOCAL vLLM server to generate responses.
    """

    def __init__(self, model_name: str):
        """
        Initializes the OpenAIGenerator with the specified model name.
        """
        # Force the model name to match your vLLM server
        # (Even if the config asks for gpt-4, we swap it here)
        self.model_name: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
        self.results: Dict = {}

    def _process_query(
            self, args: Tuple[str, List[ChatCompletionMessageParam], Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Internal method to process a single query using the LOCAL vLLM server.
        """
        q_id, messages, kwargs = args
        
        # Extract parameters (Qwen supports these, so we can keep them)
        temperature = kwargs.pop("temperature", 0.1) # Low temp for finance math
        top_p = kwargs.pop("top_p", 1.0)
        stream = kwargs.pop("stream", False)
        max_tokens = kwargs.pop("max_tokens", 2048)
        presence_penalty = kwargs.pop("presence_penalty", 0.0)
        frequency_penalty = kwargs.pop("frequency_penalty", 0.0)

        # --- THE CRITICAL MODIFICATION ---
        # We connect to localhost:8000 instead of openai.com
        client = openai.OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="sk-dummy"
        )
        # ---------------------------------

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            return q_id, response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error processing query {q_id}: {e}")
            return q_id, ""

    def generation(
            self,
            messages: Dict[str, List[Dict[str, str]]],
            num_processes: int = 4,  # Reduced to 4 to prevent VRAM OOM on 5090
            **kwargs,
    ) -> Dict[str, str]:
        """
        Generate responses for the given messages using the Local model.
        """
        logger.info(
            f"Starting generation for {len(messages)} queries using {num_processes} processes..."
        )

        # Prepare arguments for multiprocessing
        query_args = [
            (q_id, cast(list[ChatCompletionMessageParam], msg), kwargs.copy())
            for q_id, msg in messages.items()
        ]

        # Use multiprocessing Pool for parallel generation
        with Pool(processes=num_processes) as pool:
            results = pool.map(self._process_query, query_args)

        # Collect results
        self.results = {q_id: content for q_id, content in results}
        logger.info(
            f"Generation completed for all queries. Collected {len(self.results)} results."
        )

        return self.results