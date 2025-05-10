from logging import getLogger
import unittest

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableSerializable

from langchain_logseq.loaders.journal_filesystem_loader import LogseqJournalFilesystemLoader
from langchain_logseq.loaders.journal_loader_input import LogseqJournalLoaderInput
from langchain_logseq.retrievers.contextualizer import RetrieverContextualizer, RetrieverContextualizerProps
from langchain_logseq.retrievers.journal_date_range_retriever import LogseqJournalDateRangeRetriever

from chat_chain.chain import ChatChain, ChatChainProps

from utils.bedrock_api import get_bedrock_client_from_environ
from utils.logging import _enable_logging


logger = getLogger(__name__)


class TestIntegChatChain(unittest.TestCase):
    def setUp(self):
        _enable_logging()
        self.bedrock_client = get_bedrock_client_from_environ()

        # pre-build chain components
        # use a low-cost Claude model for integ testing
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            # model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            model_kwargs={
                "temperature": 0.69,
            },
            streaming=True,
        )

        # set up Retriever dependencies
        # this path is available IFF using langchain-logseq as editable:
        test_journal_path = "./integ-tests/test_journals"
        loader = LogseqJournalFilesystemLoader(test_journal_path)
        contextualizer = RetrieverContextualizer(
            RetrieverContextualizerProps(
                llm=self.llm,
                prompt=(
                    "Given the user_input, and optional chat_history, create an query object based"
                    "on the schema provided, if you believe it is relevant. Do not include anything"
                    "except for the schema, serialized as JSON. Do not answer the question directly"
                ),
                output_schema=LogseqJournalLoaderInput,
                enable_chat_history=True,
            )
        )
        self.logseq_retriever = LogseqJournalDateRangeRetriever(
            contextualizer,
            loader,
        )

    def test_chat_without_retriever(self):
        """With and without chat history fed in"""
        chat_chain_props = ChatChainProps(
            chat_llm=self.llm,
            chat_prompt=(
                "You are a helpful assistant. Answer the following question based on the"
                " provided context. If you don't know the answer, just say that you don't know."
                " Don't make up an answer."
            ),
            retriever=None,
        )
        chat_chain = ChatChain(chat_chain_props)
        self.assertIsInstance(chat_chain, ChatChain)
        self.assertIsInstance(chat_chain.qa_chain, RunnableSerializable)

        history = []
        question = "What is the capital of France? Tell me only its name."
        response = chat_chain.chat(question)
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=response))
        self.assertTrue("Paris" in response)

        question = "Is the Eiffel Tower in that city?"
        response = chat_chain.chat_and_update_history(question, history)

        self.assertGreaterEqual(len(history), 4)
        logger.info(history)

    def test_chat_with_logseq_date_range_retriever(self):
        chat_chain_props = ChatChainProps(
            chat_llm=self.llm,
            chat_prompt=(
                "You are a helpful assistant. Answer the following question based on the provided context."
            ),
            retriever=self.logseq_retriever,
        )
        chat_chain = ChatChain(chat_chain_props)

        history = []
        question = "Did I wake up early on Mar 27, 2025?"
        response = chat_chain.chat_and_update_history(question, history)
        self.assertEqual(len(history), 2)

        response = chat_chain.chat_and_update_history("What did I do on the next day? Include the date.", history)
        self.assertIn("28", response)
        logger.info(response)
        
    def test_chat_streaming(self):
        chat_chain_props = ChatChainProps(
            chat_llm=self.llm,
            chat_prompt=(
                "You are a helpful assistant. Answer the following question based on the"
                " provided context. If you don't know the answer, just say that you don't know."
                " Don't make up an answer."
            ),
            retriever=None,
        )
        chat_chain = ChatChain(chat_chain_props)
        history = []
        
        gen = chat_chain.stream("What is the capital of France?", history)
        for chunk in gen:
            logger.info(chunk)
