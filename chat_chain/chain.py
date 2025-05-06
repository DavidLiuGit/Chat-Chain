from typing import Annotated, Optional, Union, Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from chat_chain.utils.bedrock_api import get_bedrock_client_from_environ
from chat_chain.utils.telemetry import _enable_logging


class ChatChainProps(BaseModel):
    chat_llm: Annotated[
        BaseLanguageModel,
        Field(
            description="The language model to use for chat. Instance of BaseLanguageModel",
        ),
    ]
    
    chat_prompt: Annotated[
        Union[str, Callable[..., str]],
        Field(
            description="Prompt to use for the LLM. Either a string or a function that returns a string.",
        ),
    ]
    
    retriever: Annotated[
        BaseRetriever,
        Field(
            description="The retriever to use to inject context into a conversation. Instance of BaseRetriever",
        ),
    ]
    



class ChatChain(Runnable):
    
    
    def __init__(
        self,
        chat_chain_props: ChatChainProps,
    ):
        self.props = chat_chain_props
    
    
    # def invoke(
    #     self,
        
    # )
