 
![AI-Powered-Python-Learning-Assistant](AI-Powered-Python-Learning-Assistant.gif)  
**Website**: [AI-Powered Python Learning Assistant](https://ai-powered-python-learning-assistant.streamlit.app/)

# AI-Powered Python Learning Assistant

This project integrates Hugging Face's HuggingChat with an online Python IDE to enhance the learning experience. Users can ask programming-related questions and run code directly from the interface.

- 🤖 **HuggingChat Integration**: This project integrates **HuggingChat** with an online **Python IDE** to enhance the learning experience.
- 🌐 **Create Account on Hugging Face**:  
  - Sign up on [Hugging Face](https://huggingface.co/welcome).
  - Use your username and password to access the application.
- 💬 **Enter Queries**:  
  - Ask questions like "Explain different ways in which I can use the print statement."
  - **HuggingChat** will generate a response based on the query.
- 🧑‍💻 **IDE Integration**:  
  - The response from **HuggingChat** can be parsed into the **Python IDE** for execution.
  - Users can also upload input data and files for processing by the IDE.
- 📥 **Download Code & Response**:  
  - After execution, the generated code and its response can be downloaded for reference.
---

### Key Features:
- 💻 **Streamlit Interface**: User-friendly web interface to interact with the model and run Python code.
- 🤖 **HuggingChat Integration**: Uses **HuggingChat** for generating Python code based on user inputs.
- 🔧 **Automatic Module Installation**: Installs required Python modules automatically.
- 💡 **Code Generation**: Displays the generated Python code in an editor for execution.
- 📁 **File Uploads & Input Handling**: Allows file uploads and input data processing.
- 🚀 **Execution Environment**: Runs generated code in a local Python environment, displaying real-time results.

---

## Installation

### 1. Create an Account on Hugging Face  
- Sign up at [Hugging Face](https://huggingface.co/welcome).

### 2. Install **Anaconda**
- Download and install Anaconda from [here](https://www.anaconda.com/products/distribution).
- After installation, create a new Conda environment using the provided `LearnPythonWithAI.yml` file. Run the following command in your terminal:
  ```bash
  conda env create -f LearnPythonWithAI.yml
  ```
- Once the environment is created, activate it using:
  ```bash
  conda activate LearnPythonWithAI
  ```

### 3. Run the Application
- After all dependencies are installed, run the application using:
  ```bash
  streamlit run AI-Powered-Python-Learning-Assistant.py
  ```
 
Feel free to reach out with any issues or for further assistance.
```
 
