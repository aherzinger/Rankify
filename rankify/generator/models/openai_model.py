from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.prompt_generator import PromptGenerator
from rankify.utils.api.litellmclient import LitellmClient
from rankify.utils.api.openaiclient import OpenaiClient

class OpenAIModel(BaseRAGModel):
    """
    **OpenAI Model** for Retrieval-Augmented Generation (RAG).

    This class integrates OpenAI's GPT models for text generation in a RAG pipeline. 
    It uses the OpenAI API to generate responses based on input prompts.

    Attributes:
        model_name (str): Name of the OpenAI model (e.g., "gpt-3.5-turbo").
        api_keys (list): List of API keys for authenticating with OpenAI.
        prompt_generator (PromptGenerator): Instance for generating prompts.
        client (OpenaiClient): Client for interacting with the OpenAI API.

    Notes:
        - This model uses OpenAI's GPT models for text generation.
        - It supports additional parameters like `max_tokens` and `temperature`.
    """
    def __init__(self, model_name: str, api_keys: list, prompt_generator: PromptGenerator, base_url: str = None):
        """
        Initialize the OpenAIModel with the OpenaiClient.

        :param model_name: Name of the OpenAI model (e.g., "gpt-3.5-turbo").
        :param api_keys: List of API keys for OpenAI.
        :param prompt_generator: Instance of PromptGenerator for generating prompts.
        :param base_url: Optional base URL for the OpenAI API.
        """
        self.model_name = model_name
        self.prompt_generator = prompt_generator
        
        self.client = OpenaiClient(keys=api_keys, base_url=base_url)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using OpenAI's API.

        :param prompt: The input prompt for the model.
        :param kwargs: Additional parameters for the OpenAI API call.
        :return: The generated response as a string.
        """
        # Todo: use this later -> Generate the prompt using the prompt generator
        #full_prompt = self.prompt_generator.generate_prompt(prompt)

        # Set default parameters for the OpenAI API call
        kwargs.setdefault("model", self.model_name)
        kwargs.setdefault("max_tokens", 128)
        kwargs.setdefault("temperature", 0.7)

        # Call the OpenAI API using the OpenaiClient
        response = self.client.chat(messages=[{"role": "user", "content": prompt}], return_text=True, **kwargs)
        return response