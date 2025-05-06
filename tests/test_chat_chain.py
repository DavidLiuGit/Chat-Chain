import unittest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from chat_chain.chain import ChatChain

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

if __name__ == "__main__":
    unittest.main()
