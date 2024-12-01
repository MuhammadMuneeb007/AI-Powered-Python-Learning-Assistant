import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from hugchat import hugchat
from hugchat.login import Login
from fpdf import FPDF
import PyPDF2
import docx
from io import BytesIO
import streamlit as st
import subprocess
import tempfile
import os
from datetime import datetime
from streamlit_ace import st_ace
import uuid
import ast
import re
import streamlit as st
import subprocess
import tempfile
import os
from datetime import datetime
from streamlit_ace import st_ace
import uuid
import ast



# Function to extract Python code or general code blocks from the response
def extract_code_from_response(response_text):
    """
    Extract code from the model response between triple backticks (```) or 
    ```python backticks, handling multiple blocks.
    
    Args:
        response_text (str): The model response text containing code blocks.
    
    Returns:
        str: Extracted code blocks joined by double newlines, or a message if no code is found.
    """
    try:
        # Regular expression to match code blocks
        pattern = r"```(?:python)?(.*?)```"  # Matches both ``` and ```python
        code_blocks = []

        # Find all matches for the pattern
        matches = re.finditer(pattern, response_text, re.DOTALL)

        # Extract and clean each matched code block
        for match in matches:
            code_block = match.group(1).strip()
            code_blocks.append(code_block)

        # If no code blocks are found, return a message
        if not code_blocks:
            return "No code blocks found in the response."

        # Join all code blocks with double newlines for clarity
        return "\n\n".join(code_blocks)
    
    except Exception as e:
        return f"Error extracting code: {str(e)}"
    
# HugChat authentication and initialization
@st.cache_resource
def initialize_hugchat(username, password):
    """Authenticate with HugChat and return a chatbot instance."""
    cookie_path_dir = "./cookies/"  # Directory to store cookies
    sign = Login(username, password)
    cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    conversation_id = chatbot.new_conversation()
    #chatbot.change_conversation(conversation_id)
    return chatbot
# Streamlit page setup


import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AI-Powered Python Learning Assistant", layout="wide", initial_sidebar_state="expanded")

# Set custom CSS for theme
st.markdown("""
    <style>
        /* Body styling for a clean, modern look */
        body {
            background-color: #F4F6F9;
            color: #333333;
            font-family: 'Arial', sans-serif;
        }

        /* Header styling */
        h1 {
            font-family: 'Georgia', serif;
            color: #1e3a8a;
            text-align: center;
        }

        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f7fafc;
            color: #333333;
            padding: 10px;
            font-size: 14px;
        }

        /* Custom chat bubbles */
        .chat-bubble {
            padding: 12px 16px;
            border-radius: 20px;
            margin-bottom: 10px;
            max-width: 70%;
            font-size: 16px;
            line-height: 1.5;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }

        /* User bubble */
        .user-bubble {
            background-color: #e2f0fc;
            color: #1e3a8a;
            margin-left: 10px;
            border-radius: 20px 20px 0 20px;
            align-self: flex-start;
        }

        /* Assistant bubble */
        .assistant-bubble {
            background-color: #d1e7dd;
            color: #0f5132;
            margin-right: 10px;
            border-radius: 20px 20px 20px 0;
            align-self: flex-end;
        }

        /* Button and Input styling */
        .stButton>button {
            background-color: #1e3a8a;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }

        .stButton>button:hover {
            background-color: #374151;
        }

        .stTextInput>div>input {
            font-size: 14px;
            border-radius: 8px;
            padding: 12px;
            width: 100%;
        }
        
        /* File uploader styles */
        .stFileUploader {
            background-color: #f7fafc;
            padding: 12px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    /* Adjust the sidebar width */
    .css-1d391kg {  /* This targets the sidebar container in Streamlit */
        width: 200px; /* Set the desired width */
    }

    .css-1d391kg .sidebar-content {
        background-color: #f7fafc;
        color: #333333;
        padding: 10px;
        font-size: 14px;
    }

    /* Adjust the main content to align properly with the reduced sidebar */
    .css-1kyxreq {  /* This targets the main container */
        margin-left: 220px; /* Align with the new sidebar width */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App Header and Introduction
st.markdown("<h1>AI-Powered Python Learning Assistant</h1>", unsafe_allow_html=True)

 
with st.sidebar:
    st.markdown("<h3>📚 About the App</h3>", unsafe_allow_html=True)
    st.write("""
        This application leverages HugChat's API to offer a ChatGPT-like experience. It is designed to help students learn Python.
        **Key Features:**
        - 🤖 Chat with an AI assistant.
        - 📄 Download your conversation as a PDF.
        - 🧹 Clear the conversation at any time.

        **How to Use:**
        1. 👤 Enter your username and password to login. Visit [HuggingFace Join Page](https://huggingface.co/join).
        2. 💬 Start chatting with the assistant.
        3. 📥 Download the conversation as a PDF or 🧹 clear it at any time.
    """)


# Sidebar login form
with st.sidebar:
    username = st.text_input("Username", key="username", type="default", label_visibility="collapsed")
    password = st.text_input("Password", key="password", type="password", label_visibility="collapsed")

    if st.button("HugChat Login"):
        if username and password:
            try:
                st.session_state.chatbot = initialize_hugchat(username, password)
                st.success("Login successful!")
            except Exception as e:
                st.error(f"Login failed: {e}")


# Create 3 columns (Right for Chat, Left for Code Editor, Rest for Output and File Upload)
col1, col2, col3 = st.columns([2, 5, 5])  # Right for chat, left for code editor, rest for output and file upload

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llama_response" not in st.session_state:
    st.session_state.llama_response = ""

if "extracted_code" not in st.session_state:
    st.session_state.extracted_code = ""

if "model_code" not in st.session_state:
    st.session_state.model_code = ""

# Left Column (Input Data and File Uploads)
with col1:
    # Input for stdin
    st.subheader("📥 Input Data for Script")
    stdin_data = st.text_area("Provide input data:", height=150, placeholder="Input data for the code...")
   

    # Initialize session state for session_dir if it doesn't exist
    if "session_dir" not in st.session_state:
        st.session_state["session_dir"] = tempfile.mkdtemp()  # Create a temporary directory for session files

    # File upload section
    uploaded_files = st.file_uploader(
        "📂 Upload Files for Your Code (accessible via their paths)",
        accept_multiple_files=True,
    )

    uploaded_files_dict = {}

    # Save uploaded files with unique names in the session directory
    if uploaded_files:
        st.subheader("📁 Uploaded Files")
        for uploaded_file in uploaded_files:
            unique_name = f"{uuid.uuid4()}_{uploaded_file.name}"
            file_path = os.path.join(st.session_state["session_dir"], unique_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_files_dict[uploaded_file.name] = file_path

            # Display file details
            st.markdown(f"**File Name:** `{uploaded_file.name}`")
            st.markdown(f"**Server Path:** `{file_path}`")
    else:
        st.info("No files uploaded yet.")
import matplotlib.pyplot as plt
# Left Column (Code Editor and Output)

def install_packages(code):
    # Parse the code to extract imported modules
    tree = ast.parse(code)
    imports = set()

    # Find import statements
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)

    # Prepare the install commands using pip
    for package in imports:
        try:
            subprocess.run(["pip", "install", package], check=True)
        except subprocess.CalledProcessError:
            st.error(f"Failed to install package: {package}")

with col2:
    st.subheader("✍️ Write Your Python Code")

    # Button to extract content from the Ollama model and insert it into the editor
    if st.button("💬 Get Code from HugChat Output"):
        st.session_state["code"] = st.session_state.messages[-1]["content"]
        st.session_state.extracted_code = extract_code_from_response(st.session_state.messages[-1]["content"])

    # Create a row for theme, font size, and editor height in one line
    theme_col, font_col, height_col = st.columns([3, 2, 2])  # Adjusting column width to align properly

    with theme_col:
        theme = st.selectbox(
            "🎨 Select Theme",
            ["monokai", "dracula", "solarized_light", "github", "xcode", "tomorrow_night"]
        )

    with font_col:
        font_size = st.selectbox("🔠 Select Font Size", [10, 12, 14, 16, 18, 20, 24, 28, 36],index=4 )

    with height_col:
        editor_height = st.selectbox("📏 Select Editor Height", [200, 400, 600, 800],index=1)

    # Display the extracted code in the editor
    code = st_ace(
        value=st.session_state.extracted_code,  # Show the extracted code
        language="python",  # Set language as Python for syntax highlighting
        theme=theme,  # Theme selected by the user
        font_size=font_size,  # Font size selected by the user
        height=editor_height,  # User-defined height
        auto_update=True,  # Automatically update value when editing
        placeholder=st.session_state.extracted_code,
        show_gutter=True,
        wrap=True,
    )

    # Store the modified code in session state
    st.session_state["code"] = code
    

    # Run Code button
    if st.button("▶️ Run Code"):
        code = st.session_state.get("code", "")
        if not code.strip():
            st.error("❌ No code provided! Please write some Python code.")
        else:
            # Save the code to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
                temp_file.write(code)
                temp_file_name = temp_file.name
            try:
                install_packages(code)
            except:
                pass

            try:
                # Running the Python code directly
                result = subprocess.run(
                    ["python", temp_file_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                # Display the results
                st.subheader("📊 Execution Output")
                st.text_area("Output", result.stdout, height=200)
                if result.stderr:
                    st.text_area("⚠️ Errors", result.stderr, height=200)

                # Attempt to display plots, dataframes, or other outputs
                try:
                    # Check if the plot generated is static
                    if 'plt' in globals() and isinstance(plt.gcf(), plt.Figure):
                        st.pyplot(plt.gcf())  # Display static matplotlib plot
                        plt.clf()  # Clear the figure after displaying it

                    # Check if the code generates a Plotly plot (interactive)
                    elif 'fig' in globals() and isinstance(fig, go.Figure):
                        st.plotly_chart(fig)  # Display Plotly interactive plot
                except Exception as plot_error:
                    st.error(f"❌ Error in displaying plot: {plot_error}")

                # Attempt to display any generated DataFrames (for example)
                try:
                    # If the code generates a DataFrame 'df'
                    if 'df' in globals() and isinstance(df, pd.DataFrame):
                        st.subheader("📊 Generated DataFrame")
                        st.dataframe(df)
                except Exception as df_error:
                    st.error(f"❌ Error in displaying dataframe: {df_error}")

                # Log the execution
                if "execution_log" not in st.session_state:
                    st.session_state.execution_log = []

                st.session_state.execution_log.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "code": code,
                    "output": result.stdout,
                    "error": result.stderr,
                })

            except subprocess.TimeoutExpired:
                st.error("❌ Execution timed out. Ensure your code doesn't contain infinite loops.")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")
            finally:
                # Cleanup the temporary code file
                if os.path.exists(temp_file_name):
                    os.remove(temp_file_name)
    
    if st.button("▶️ Download Code"):
        code = st.session_state.get("code", "")

        if not code.strip():
            st.error("❌ No code available to download.")
        else:
            # Create a temporary file with the code content
            st.download_button(
                label="Download Python Code",  # Label for the download button
                data=code,  # The actual code to be downloaded
                file_name="script.py",  # The file name when downloading
                mime="text/plain"  # Mime type for a plain text file
        )


    # Display code and model response
    st.write("## 📝 Model code:")
    st.code(st.session_state.extracted_code, language="python")

    st.write("## 📑 Model Response:")
    st.text_area(label="Model Response", value=st.session_state.llama_response, height=200)

import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import markdown
import re
# Right Column (Chatbox for user input and Ollama model interaction)
with col3:
    # Ensure HugChat chatbot is initialized
    chatbot = st.session_state.get("chatbot")

    # Memory for conversation
    memory = ConversationBufferMemory()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat and File Upload Section
    st.markdown("<hr>", unsafe_allow_html=True)
    file_uploaded = False
    file_content = ""
    

    # Display the input for chat interaction
    if file_uploaded and file_content:
        user_input = st.text_area("Ask something related to the file...", value=file_content)
    else:
        user_input = st.chat_input("Type your message...")

    # Process and display chat messages
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if chatbot:
            assistant_response = chatbot.chat(user_input)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response['text']})
        else:
            st.error("Chatbot is not initialized. Please log in via the sidebar.")
        
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble user-bubble">🧑 <strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            st.markdown(f'<div class="chat-bubble assistant-bubble">🤖 <strong>Assistant:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

    # Clear conversation button
    if st.button("Clear Conversation", key="clear_button"):
        st.session_state.messages = []


    # Function to clean text for PDF compatibility and convert Markdown
    def clean_text_for_pdf(text):
        # Replace unsupported characters with simple ASCII equivalents or remove them
        cleaned_text = text.replace('’', "'").replace('“', '"').replace('”', '"').replace('–', '-')
        return cleaned_text
    import streamlit as st
    import markdown
    import pdfkit
    from io import BytesIO

    # Function to clean text for PDF compatibility
    def clean_text_for_pdf(text):
        # Replace unsupported characters with simple ASCII equivalents or remove them
        cleaned_text = text.replace('’', "'").replace('“', '"').replace('”', '"').replace('–', '-')
        return cleaned_text

    # Function to convert Markdown content to HTML
    def markdown_to_html(md_text):
        # Convert markdown to HTML
        html_text = markdown.markdown(md_text)
        return html_text

    # Download conversation as PDF
    if st.button("Download Conversation as PDF", key="download_button"):
        if len(st.session_state.messages) > 0:

            # Create HTML content for the PDF
            html_content = "<html><body style='font-family: Arial, sans-serif;'>"
            
            # Loop through each message and convert it to HTML
            for msg in st.session_state.messages:
                role = "You" if msg["role"] == "user" else "Assistant"
                html_content += f"<h3>{role}:</h3><p>{markdown_to_html(msg['content'])}</p>"
            
            html_content += "</body></html>"

            # Convert the HTML content to a PDF using pdfkit
            pdf_output = pdfkit.from_string(html_content, False)

            # Provide download button for the PDF
            st.download_button("Download PDF", pdf_output, "conversation.pdf", mime="application/pdf")
        else:
            st.warning("No conversation to download.")




# Display execution history in a collapsible section
if "execution_log" in st.session_state and st.session_state.execution_log:
    st.subheader("📜 Execution History")
    for log in st.session_state.execution_log[-5:]:  # Display the last 5 executions
        st.markdown(f"**🕒 Time:** {log['time']}")
        st.markdown("**📋 Code:**")
        st.code(log["code"], language="python")
        if log["stdin"]:
            st.markdown("**📥 Input Data (stdin):**")
            st.code(log["stdin"], language="python")
        st.markdown("**📝 Output:**")
        st.code(log["output"], language="python")
        if log["error"]:
            st.markdown("**⚠️ Errors:**")
            st.code(log["error"], language="python")
