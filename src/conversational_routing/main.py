#!/usr/bin/env python
from pydantic import BaseModel
from typing import List
import json

from crewai.flow import Flow, start, persist
from litellm import completion

from src.conversational_routing.crews.assistant_crew.assistant_crew import AssistantCrew


class ChatState(BaseModel):
    current_message: str = ""
    conversation_history: List[dict] = []
    conversation_summary: str = ""  # Running summary of older context
    summary_threshold: int = 8      # Summarize when history exceeds this
    keep_recent: int = 4            # Keep this many recent messages in full

@persist()
class ChatFlow(Flow[ChatState]):
    
    def _summarize_conversation(self) -> None:
        """
        Summarize older messages when conversation history exceeds threshold.
        Keeps recent messages in full detail while compressing older context.
        """
        history_length = len(self.state.conversation_history)
        
        if history_length <= self.state.summary_threshold:
            return  # No need to summarize yet
        
        # Calculate how many messages to summarize
        messages_to_summarize = history_length - self.state.keep_recent
        if messages_to_summarize <= 0:
            return
        
        # Extract older messages to summarize
        older_messages = self.state.conversation_history[:messages_to_summarize]
        
        # Format messages for summarization
        formatted_messages = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in older_messages
        ])
        
        # Include existing summary if present
        existing_context = ""
        if self.state.conversation_summary:
            existing_context = f"Previous summary: {self.state.conversation_summary}\n\n"
        
        # Use lightweight LLM call for summarization
        summary_prompt = f"""Summarize the following conversation context concisely, preserving key facts, user preferences, and important details mentioned. Keep it under 200 words.

{existing_context}New messages to incorporate:
{formatted_messages}

Concise summary:"""

        response = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        # Update summary and trim conversation history
        self.state.conversation_summary = response.choices[0].message.content
        self.state.conversation_history = self.state.conversation_history[messages_to_summarize:]
    
    @start()
    def answer_message(self):
        # Check if we need to summarize before processing
        self._summarize_conversation()
        
        # Here define the crew that will respond to the user message
        assistant_crew = AssistantCrew().crew().kickoff({
            "current_message": self.state.current_message,
            "conversation_history": self.state.conversation_history,
            "conversation_summary": self.state.conversation_summary
        })

        response = assistant_crew.raw

        self.state.conversation_history.append({"role": "user", "content": self.state.current_message})
        self.state.conversation_history.append({"role": "assistant", "content": response})

        return json.dumps({
            "response": response,
            "id": self.state.id
        })
    
def kickoff():
    chat_flow = ChatFlow()
    chat_flow.kickoff(inputs={})

def plot():
    chat_flow = ChatFlow()
    chat_flow.plot()

if __name__ == "__main__":
    kickoff()
