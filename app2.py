from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is missing. Please ensure it is set in the .env file.")

# Initialize Chat Model
chat_model = ChatOpenAI(temperature=0.7, model="gpt-4o-mini", openai_api_key=openai_api_key)

# Define Prompt Template
prompt_template = ChatPromptTemplate.from_template("""
You are an intelligent recipe assistant. Respond to user queries as follows:
- Provide detailed, helpful, and user-friendly responses to human food recipe-related questions.
- Include cooking procedures, ingredients, alternatives, cuisine suggestions, and cultural insights when relevant.
- If a user asks about animal food recipes, politely inform them that you only assist with human food recipes.
- If a user asks about their previous steps (e.g., "What are my previous steps?"), summarize the chat history so far.
- If a user asks an unrelated question (e.g., weather or sports), politely inform them that you can only assist with recipes.

Conversation so far:
{history}

User input: {input}

Your response:
""")

# Set up memory for context
memory = ConversationBufferMemory()

# Create a conversation chain
conversation = ConversationChain(
    llm=chat_model,
    prompt=prompt_template,
    memory=memory
)

# FastAPI App Initialization
app = FastAPI()

# Define Pydantic model for user input
class UserInput(BaseModel):
    message: str

@app.post("/chat/")
async def chat(user_input: UserInput):
    try:
        # Handle user input
        message = user_input.message.strip()

        # Handle special query about previous steps
        if "previous steps" in message.lower():
            # Extract and summarize the chat history from memory
            chat_history = memory.chat_memory.messages
            if not chat_history:
                return {"response": "There are no previous steps yet."}

            history_summary = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history])
            return {"response": f"Here are your previous steps so far:\n\n{history_summary}"}

        # Check for animal food-related queries
        animal_keywords = ["dog", "cat", "animal", "pet"]
        if any(keyword in message.lower() for keyword in animal_keywords):
            return {"response": "I only assist with human food recipes. Please ask about cooking or recipes for humans!"}

        # Generate chatbot response
        response = conversation.run(input=message)
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

