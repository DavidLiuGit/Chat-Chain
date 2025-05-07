from logging import getLogger
import unittest

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage

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
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            model_kwargs={
                "temperature": 0.3,
            },
        )

        # set up Retriever dependencies
        # this path is available IFF using langchain-logseq as editable:
        test_journal_path = "./venv/src/langchain-logseq/tests/loaders/test_journals"
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
        self.retriever = LogseqJournalDateRangeRetriever(
            contextualizer,
            loader,
        )

    def test_chat_without_retriever(self):
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
        response = chat_chain.chat("What is the capital of France?")
        logger.info(response)
