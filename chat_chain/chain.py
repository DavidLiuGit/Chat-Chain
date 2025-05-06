from typing import Annotated, Optional, Union, Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
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
    



class ChatChain:
    
    
    def __init__(
        self,
        chat_chain_props: ChatChainProps,
    ):
        self.props = chat_chain_props
        
    
    def chat(
        self,
        user_input: str,
        chat_history: list[BaseMessage]
    ) -> str:
        pass
    
    
    @staticmethod
    def build_structured_chat_history(unstructured_chat_history: list[tuple[str, str]]) -> list[BaseMessage]:
        """
        Convert a list of chat messages to a structured `list[BaseMessage]`, required by the `chat` method.
        Each tuple in `unstructured_chat_history` is expected to have the format (actor, message_str). e.g.:
        ```
        unstructured_chat_history = [
            ("human", "What did I do on my birthday?"),
            ("system", "The user stayed at home all day and watched TV."),  # system-msgs are optional
            ("ai", "Sounds like you had a great time on your birthday!"),
        ]
        ```
        """
        output: list[BaseMessage] = []
        for input_msg in unstructured_chat_history:
            agent = input_msg[0].lower()
            if agent == "human":
                output.append(HumanMessage(content=input_msg[1]))
            elif agent == "ai":
                output.append(AIMessage(content=input_msg[1]))
            elif agent == "system":
                output.append(SystemMessage(content=input_msg[1]))
            # if the agent is not one of the above, ignore it
        return output
    