import unittest
from unittest.mock import Mock, patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from chat_chain.chain import ChatChain, ChatChainProps
from chat_chain.utils.chat_prompt_template import (
    QA_CHAIN_LLM_PROMPT_TEPLATE,
    QA_CHAIN_LLM_PROMPT_WITH_RETRIEVER_TEMPLATE,
)


class TestChatChain(unittest.TestCase):

    def test_build_structured_chat_history_empty(self):
        """Test with an empty chat history."""
        unstructured_history = []
        structured_history = ChatChain.build_structured_chat_history(unstructured_history)
        self.assertEqual(len(structured_history), 0)
        self.assertIsInstance(structured_history, list)

    def test_build_structured_chat_history_single_message(self):
        """Test with a single message in chat history."""
        unstructured_history = [("human", "Hello, how are you?")]
        structured_history = ChatChain.build_structured_chat_history(unstructured_history)

        self.assertEqual(len(structured_history), 1)
        self.assertIsInstance(structured_history[0], HumanMessage)
        self.assertEqual(structured_history[0].content, "Hello, how are you?")

    def test_build_structured_chat_history_multiple_messages(self):
        """Test with multiple messages of different types."""
        unstructured_history = [
            ("human", "What did I do on my birthday?"),
            ("system", "The user stayed at home all day and watched TV."),
            ("ai", "Sounds like you had a great time on your birthday!"),
        ]
        structured_history = ChatChain.build_structured_chat_history(unstructured_history)

        self.assertEqual(len(structured_history), 3)
        self.assertIsInstance(structured_history[0], HumanMessage)
        self.assertIsInstance(structured_history[1], SystemMessage)
        self.assertIsInstance(structured_history[2], AIMessage)

        self.assertEqual(structured_history[0].content, "What did I do on my birthday?")
        self.assertEqual(structured_history[1].content, "The user stayed at home all day and watched TV.")
        self.assertEqual(structured_history[2].content, "Sounds like you had a great time on your birthday!")

    def test_build_structured_chat_history_case_insensitive(self):
        """Test that actor types are case-insensitive."""
        unstructured_history = [
            ("HUMAN", "Hello"),
            ("AI", "Hi there"),
            ("System", "This is a system message"),
        ]
        structured_history = ChatChain.build_structured_chat_history(unstructured_history)

        self.assertEqual(len(structured_history), 3)
        self.assertIsInstance(structured_history[0], HumanMessage)
        self.assertIsInstance(structured_history[1], AIMessage)
        self.assertIsInstance(structured_history[2], SystemMessage)

    def test_build_structured_chat_history_ignore_unknown(self):
        """Test that unknown actor types are ignored."""
        unstructured_history = [
            ("human", "Hello"),
            ("unknown", "This should be ignored"),
            ("ai", "Hi there"),
        ]
        structured_history = ChatChain.build_structured_chat_history(unstructured_history)

        self.assertEqual(len(structured_history), 2)
        self.assertIsInstance(structured_history[0], HumanMessage)
        self.assertIsInstance(structured_history[1], AIMessage)
        self.assertEqual(structured_history[0].content, "Hello")
        self.assertEqual(structured_history[1].content, "Hi there")


class TestBuildQuestionAndAnswerChain(unittest.TestCase):

    def setUp(self):
        # Create mocks for dependencies
        self.mock_llm = Mock(spec=BaseLanguageModel)
        self.mock_retriever = Mock(spec=BaseRetriever)
        self.chat_prompt_str = "This is a test prompt"
        self.chat_prompt_callable = lambda: "This is a callable test prompt"

    def test_build_qa_chain_no_retriever(self):
        """Test building the Q&A chain without a retriever."""
        # Create ChatChainProps without a retriever
        props = ChatChainProps(chat_llm=self.mock_llm, chat_prompt=self.chat_prompt_str, retriever=None)

        # Create ChatChain instance
        chat_chain = ChatChain(props)

        # Verify the chain was built correctly
        self.assertIsInstance(chat_chain.qa_chain, Runnable)
        self.assertIsInstance(chat_chain.qa_prompt_template, PromptTemplate)

        # Verify the prompt template has the correct variables
        self.assertEqual(sorted(chat_chain.qa_prompt_template.input_variables), sorted(["chat_history", "user_input"]))

        # Verify the prompt template uses the correct template string
        self.assertEqual(chat_chain.qa_prompt_template.template, QA_CHAIN_LLM_PROMPT_TEPLATE)

        # Verify the chain input has the correct keys
        self.assertEqual(sorted(chat_chain.chain_input.keys()), sorted(["chat_history", "user_input"]))

    def test_build_qa_chain_with_retriever(self):
        """Test building the Q&A chain with a retriever."""
        # Create ChatChainProps with a retriever
        props = ChatChainProps(chat_llm=self.mock_llm, chat_prompt=self.chat_prompt_str, retriever=self.mock_retriever)

        # Create ChatChain instance
        chat_chain = ChatChain(props)

        # Verify the chain was built correctly
        self.assertIsInstance(chat_chain.qa_chain, Runnable)
        self.assertIsInstance(chat_chain.qa_prompt_template, PromptTemplate)

        # Verify the prompt template has the correct variables
        self.assertEqual(
            sorted(chat_chain.qa_prompt_template.input_variables),
            sorted(["chat_history", "user_input", "document_context"]),
        )

        # Verify the prompt template uses the correct template string
        self.assertEqual(chat_chain.qa_prompt_template.template, QA_CHAIN_LLM_PROMPT_WITH_RETRIEVER_TEMPLATE)

        # Verify the chain input has the correct keys
        self.assertEqual(
            sorted(chat_chain.chain_input.keys()), sorted(["chat_history", "user_input", "document_context"])
        )

    def test_build_qa_chain_with_string_prompt(self):
        """Test building the Q&A chain with a string prompt."""
        props = ChatChainProps(chat_llm=self.mock_llm, chat_prompt=self.chat_prompt_str, retriever=None)

        chat_chain = ChatChain(props)

        # Verify the prompt was correctly set in the partial variables
        self.assertEqual(chat_chain.qa_prompt_template.partial_variables["prompt"], self.chat_prompt_str)

    def test_build_qa_chain_with_callable_prompt(self):
        """Test building the Q&A chain with a callable prompt."""
        props = ChatChainProps(chat_llm=self.mock_llm, chat_prompt=self.chat_prompt_callable, retriever=None)

        chat_chain = ChatChain(props)

        # Verify the prompt was correctly set in the partial variables
        self.assertEqual(chat_chain.qa_prompt_template.partial_variables["prompt"], self.chat_prompt_callable())


if __name__ == "__main__":
    unittest.main()
