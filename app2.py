import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is missing. Please ensure it is set in the .env file.")
openai.api_key = openai_api_key

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

# Function to save chat history
def save_chat_history(chat_history, folder_path="chat_data"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, "chat_history.txt")
    with open(file_path, "w") as file:
        for entry in chat_history:
            user_message = entry[0]
            bot_response = entry[1]
            file.write(f"User: {user_message}\nResponse: {bot_response}\n\n")
    return "Chat history saved successfully!"

# Function to handle user input
def recipe_chatbot(user_input):
    try:
        # Handle special query about previous steps
        if "previous steps" in user_input.lower():
            # Extract and summarize the chat history from memory
            chat_history = memory.chat_memory.messages
            if not chat_history:
                return "There are no previous steps yet."

            history_summary = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history])
            return f"Here are your previous steps so far:\n\n{history_summary}"

        # Check for animal food-related queries
        animal_keywords = ["dog", "cat", "animal", "pet"]
        if any(keyword in user_input.lower() for keyword in animal_keywords):
            return "I only assist with human food recipes. Please ask about cooking or recipes for humans!"

        # Check for greeting messages
        greetings = ["hi", "hello", "hey"]
        if user_input.strip().lower() in greetings:
            return "Hello! How can I assist you with recipes today?"

        # Generate chatbot response
        response = conversation.run(input=user_input)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Terminal-based chatbot
def main():
    print("Welcome to the Intelligent Recipe Chatbot!")
    print("Type your recipe-related questions below (or type 'exit' to quit).")
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye! Have a great day!")
            break
        
        response = recipe_chatbot(user_input)
        print(f"Bot: {response}")
        chat_history.append((user_input, response))
        
        # Ask to save chat history after each response
        save_option = input("Do you want to save the chat history? (yes/no): ").strip().lower()
        if save_option == "yes":
            save_status = save_chat_history(chat_history)
            print(save_status)

if __name__ == "__main__":
    main()
