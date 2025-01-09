import os
import gradio as gr
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

# Chatbot function
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

# Gradio Interface
with gr.Blocks(css="""
    .main-container { 
        max-width: 800px; 
        margin: 0 auto; 
        padding: 20px; 
        font-family: Arial, sans-serif; 
        border: 1px solid #ddd; 
        border-radius: 10px; 
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        background-color: #fefefe; 
    }
    .title { 
        text-align: center; 
        font-size: 28px; 
        font-weight: bold; 
        color: #4CAF50; 
        margin-bottom: 10px; 
    }
    .subtitle { 
        text-align: center; 
        font-size: 16px; 
        margin-bottom: 20px; 
        color: #555; 
    }
    .button { 
        font-size: 14px; 
        padding: 5px 15px; 
        border-radius: 5px; 
        border: none; 
        cursor: pointer; 
        margin-right: 10px; 
    }
    .submit-btn { 
        background-color: #4CAF50; 
        color: white; 
    }
    .submit-btn:hover { 
        background-color: #45a049; 
    }
    .clear-btn { 
        background-color: #f44336; 
        color: white; 
    }
    .clear-btn:hover { 
        background-color: #e53935; 
    }
    .save-btn { 
        background-color: #2196F3; 
        color: white; 
    }
    .save-btn:hover { 
        background-color: #1976D2; 
    }
    .chatbox-container {
        margin-bottom: 15px; 
    }
    .chatbox {
        background-color: #f9f9f9; 
        border-radius: 8px; 
        padding: 10px; 
        border: 1px solid #ddd; 
        height: 300px; 
        overflow-y: auto; 
    }
    .textbox { 
        width: 100%; 
        margin-bottom: 10px; 
    }
""") as interface:
    with gr.Group(elem_classes=["main-container"]):
        gr.Markdown(
            """
            <div class="title">🍽️ Intelligent Recipe Chatbot</div>
            <div class="subtitle">Ask anything about recipes, cooking, or food!</div>
            """
        )

        with gr.Row(elem_classes=["chatbox-container"]):
            chatbox = gr.Chatbot(label="Chat", elem_classes=["chatbox"])

        user_input = gr.Textbox(
            placeholder="Type your question here...",
            label="Your Message",
            lines=2,
            elem_classes=["textbox"],
        )

        with gr.Row():
            submit_button = gr.Button(
                "Submit",
                elem_classes=["button", "submit-btn"],
            )
            clear_button = gr.Button(
                "Clear Chat",
                elem_classes=["button", "clear-btn"],
            )
            flag_button = gr.Button(
                "Save Data",
                elem_classes=["button", "save-btn"],
            )

        output_message = gr.Textbox(
            label="Save Confirmation", interactive=False, visible=True
        )

    # Define button functionality
    def submit_message(chat_history, input_text):
        # Check for unrelated queries
        unrelated_keywords = ["weather", "sports", "news", "jokes", "games"]
        if any(keyword in input_text.lower() for keyword in unrelated_keywords):
            response = "I can only assist with recipe-related questions. Please ask about cooking or recipes!"
            chat_history.append((input_text, response))
            return chat_history, ""

        # Generate recipe-related responses
        response = recipe_chatbot(input_text)
        chat_history.append((input_text, response))
        return chat_history, ""

    def save_data(chat_history):
        return save_chat_history(chat_history, folder_path="chat_data")

    submit_button.click(submit_message, [chatbox, user_input], [chatbox, user_input])

    clear_button.click(lambda: [], [], chatbox)
    flag_button.click(save_data, [chatbox], output_message)

# Launch the interface
interface.launch()
