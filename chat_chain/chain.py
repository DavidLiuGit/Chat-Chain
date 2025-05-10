from logging import getLogger
from typing import Annotated, Any, Callable, Iterator, Literal, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from chat_chain.utils.telemetry import _enable_logging
from chat_chain.utils.chat_prompt_template import (
    QA_CHAIN_LLM_PROMPT_TEPLATE,
    QA_CHAIN_LLM_PROMPT_WITH_RETRIEVER_TEMPLATE,
)


logger = getLogger(__name__)


class ChatChainProps(BaseModel):
    chat_llm: Annotated[
        BaseLanguageModel,
        Field(
            description="The language model to use for chat. Instance of BaseLanguageModel",
        ),
    ]

    chat_prompt: Annotated[
        str | Callable[..., str],
        Field(
            description=(
                "Prompt to use for the LLM. Either a string or a function that returns a string. "
                "There is no need to include placeholders/templates for user input, chat history, or context. "
            ),
        ),
    ]

    retriever: Annotated[
        Optional[BaseRetriever],
        Field(
            description=(
                "(optional) The retriever to use to inject context into a conversation. Instance of BaseRetriever."
                "If provided, the chat history and latest user input will be provided the the Retriever"
            ),
        ),
    ]


class ChatChain:

    def __init__(
        self,
        chat_chain_props: ChatChainProps,
    ):
        self.props = chat_chain_props
        self.qa_chain: RunnableSerializable = self._build_question_and_answer_chain()

    def chat(self, user_input: str, chat_history: list[BaseMessage] = []) -> str:
        """
        Invoke the `ChatChain`, and return the Q&A chain's output. The process will vary based on
        the properties that this chain was initially set up with.

        Args:
            user_input (`str`): User's latest input
            chat_history (`list[BaseMessage]`, optional): Structured chat history.
                Use `ChatChain.build_structured_chat_history` to build an instance.
                Defaults to [].

        Returns:
            str: Q&A LLM's output, as a string.
        """
        return self.qa_chain.invoke(self._build_chain_input(user_input, chat_history))
    
    def achat(self, user_input: str, chat_history: list[BaseMessage] = []) -> str:
        """
        Invoke the `ChatChain`, and return the Q&A chain's output. The process will vary based on
        the properties that this chain was initially set up with.

        Args:
            user_input (`str`): User's latest input
            chat_history (`list[BaseMessage]`, optional): Structured chat history.
                Use `ChatChain.build_structured_chat_history` to build an instance.
                Defaults to [].

        Returns:
            str: Q&A LLM's output, as a string.
        """
        return self.qa_chain.ainvoke(self._build_chain_input(user_input, chat_history))
        
    def chat_and_update_history(self, user_input: str, chat_history: list[BaseMessage] = []) -> str:
        """Convenience method: calls `self.chat()` and updates the chat_history. Returns chat() output."""
        output = self.chat(user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        return output
    
    def stream(self, user_input: str, chat_history: list[BaseMessage] = []) -> Iterator[str]:
        """
        Invoke the `ChatChain`, and **stream** the Q&A chain's output. The process will vary based on
        the properties that this chain was initially set up with.

        Args:
            user_input (`str`): User's latest input
            chat_history (`list[BaseMessage]`, optional): Structured chat history.
                Use `ChatChain.build_structured_chat_history` to build an instance.
                Defaults to [].

        Returns:
            str: Q&A LLM's output, 1 *chunk* at a time.
                Use `for chunk in chain.stream()` to handle output chunks.
        """
        for chunk in self.qa_chain.stream(self._build_chain_input(user_input, chat_history)):
            yield chunk


    @staticmethod
    def build_structured_chat_history(
        unstructured_chat_history: list[tuple[str, str] | dict[str, str]]
    ) -> list[BaseMessage]:
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
        if not bool(unstructured_chat_history):
            return []

        output: list[BaseMessage] = []
        if isinstance(unstructured_chat_history[0], list):
            for input_msg in unstructured_chat_history:
                agent = input_msg[0].lower()
                if agent == "human":
                    output.append(HumanMessage(content=input_msg[1]))
                elif agent == "ai":
                    output.append(AIMessage(content=input_msg[1]))
                elif agent == "system":
                    output.append(SystemMessage(content=input_msg[1]))
                # if the agent is not one of the above, ignore it
        elif isinstance(unstructured_chat_history[0], dict):
            for input_msg in unstructured_chat_history:
                agent = input_msg["agent"].lower()
                if agent == "human":
                    output.append(HumanMessage(content=input_msg["message"]))
                elif agent == "ai":
                    output.append(AIMessage(content=input_msg["message"]))
                elif agent == "system":
                    output.append(SystemMessage(content=input_msg["message"]))
        return output

    def _build_chain_input(self, user_input: str, chat_history: list[BaseMessage] = []) -> dict[str, Any]:
        """
        Using the latest `user_input` and `chat_history`, return a dict representing the input to
        the Q&A chain. Should include `user_input`, `chat_history`, and `document_context`, if applicable.
        """
        # if a Retriever was provided, use it to attempt to inject context prior to the response
        context_documents: Optional[list[Document]] = (
            self.props.retriever.invoke({"user_input": user_input, "chat_history": chat_history})
            if self.props.retriever
            else None
        )
        return {
            "user_input": user_input,
            "chat_history": chat_history,
            "document_context": context_documents,
        }

    def _build_question_and_answer_chain(self) -> RunnableSerializable:
        """
        Build the Q&A chain that will be invoked to respond to the user's input.
        """
        # build the PromptTemplate
        self.qa_prompt_template = self._build_qa_llm_prompt_template(self.props.chat_prompt, bool(self.props.retriever))
        self.chain_input = {
            "chat_history": lambda x: x["chat_history"],
            "user_input": lambda x: x["user_input"],
            # "document_context": lambda x: x["document_context"],
        }
        if self.props.retriever and isinstance(self.props.retriever, BaseRetriever):
            self.chain_input["document_context"] = lambda x: x["document_context"]

        return (
            self.qa_prompt_template
            | (lambda x: logger.debug(f"Q&A prompt: {x}") or x)  # can enable for debugging, will not fail
            | self.props.chat_llm
            | StrOutputParser()
        )

    def _build_qa_llm_prompt_template(
        self,
        chat_prompt: str | Callable[..., str],
        retriever_available: bool = False,
    ) -> PromptTemplate:
        """
        Create a `PromptTemplate` for the Q&A chain's LLM. If a `Retriever` has been provided, create a placeholder
        for document context injection.
        """
        prompt_str = chat_prompt if isinstance(chat_prompt, str) else chat_prompt()
        self.prompt_template = PromptTemplate(
            input_variables=(
                ["chat_history", "user_input", "document_context"]
                if retriever_available
                else ["chat_history", "user_input"]
            ),
            partial_variables={
                "prompt": prompt_str,
            },
            template=(
                QA_CHAIN_LLM_PROMPT_WITH_RETRIEVER_TEMPLATE if retriever_available else QA_CHAIN_LLM_PROMPT_TEPLATE
            ),
        )
        return self.prompt_template
