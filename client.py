import chainlit as cl
from llama_deploy import LlamaDeployClient, ControlPlaneConfig
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
CONTROL_PLANE_HOST = os.getenv("CONTROL_PLANE_HOST")
CONTROL_PLANE_PORT = int(os.getenv("CONTROL_PLANE_PORT", 8000))
WORKFLOW_NAME = os.getenv("WORKFLOW_NAME")

if not all([CONTROL_PLANE_HOST, WORKFLOW_NAME]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Create a global client instance
client = LlamaDeployClient(
    ControlPlaneConfig(
        host=CONTROL_PLANE_HOST,
        port=CONTROL_PLANE_PORT
    )
)

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    # Create a session and store it
    session = client.create_session()
    cl.user_session.set("llama_session", session)
    
    await cl.Message(
        content="👋 Hello! I'm your AI assistant powered by LlamaIndex. I can help you analyze "
        "Uber and Lyft's financial data from their 2021 10-K reports. What would you like to know?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    try:
        # Get the session from user session
        session = cl.user_session.get("llama_session")
        if not session:
            # If session is not found, create a new one
            session = client.create_session()
            cl.user_session.set("llama_session", session)

        # Create a thinking message
        thinking_msg = cl.Message(content="Thinking...")
        await thinking_msg.send()

        # Start the workflow and get task_id
        task_id = session.run_nowait(
            WORKFLOW_NAME,
            input=message.content
        )

        # Stream events and update the UI with intermediate results
        reasoning_steps = []
        reasoning_text = ""
        
        for event in session.get_task_result_stream(task_id):
            # Add the event to reasoning steps if it contains reasoning information
            if "reasoning" in event:
                reasoning_steps.extend(event["reasoning"])
                # Update the UI with current reasoning
                reasoning_text = ""
                for step_num, step_info in enumerate(reasoning_steps, 1):
                    if hasattr(step_info, "thought"):
                        reasoning_text += f"\n{step_num}. Thought: {step_info.thought}\n"
                    if hasattr(step_info, "action"):
                        reasoning_text += f"   Action: {step_info.action}\n"
                    if hasattr(step_info, "observation"):
                        reasoning_text += f"   Observation: {step_info.observation}\n"
                
                # Update the thinking message with current progress
                await thinking_msg.update() #reasoning_text)

        # Get the final result
        result = session.get_task_result(task_id)
        
        # Extract the response and sources
        final_response = result.get("response", "No response generated")
        sources = result.get("sources", [])
        
        # Create elements for the final message
        elements = []
        if reasoning_text:
            elements.append(
                cl.Text(name="reasoning", content=reasoning_text, display="inline")
            )
        
        if sources:
            source_text = "\nSources:\n" + "\n".join(
                [f"- {source}" for source in sources]
            )
            elements.append(
                cl.Text(name="sources", content=source_text, display="inline")
            )

        # Send the final response
        await cl.Message(
            content=final_response,
            elements=elements
        ).send()

        # Remove the thinking message
        await thinking_msg.remove()

    except Exception as e:
        # Send error message
        await cl.Message(
            content=f"❌ An error occurred: {str(e)}"
        ).send()

if __name__ == "__main__":
    cl.run()