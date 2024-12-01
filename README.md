![AI-Powered-Python-Learning-Assistant](AI-Powered-Python-Learning-Assistant.gif)
# AI-Powered-Python-Learning-Assistant
AI-Powered Python Learning Assistant


Sample: [https://pythononlineide.streamlit.app/](https://ai-powered-python-learning-assistant.streamlit.app/)
 

## Project Description

This project integrates **Streamlit**, the **Ollama** model, and a local Python environment to dynamically generate and execute Python code. Users can interact with the **Llama2** model for code generation and execution, providing a seamless experience for both input handling and file management.

### Key Features:
- **Streamlit Interface**: A user-friendly web interface to interact with the model and execute Python code.
- **Ollama Integration**: Leverages the Llama2 model via Ollama to generate Python code based on user inputs.
- **Automatic Module Installation**: The app automatically downloads the necessary Python modules required to run the generated code.
- **Code Generation**: After receiving a response from the Llama2 model, the code is extracted and displayed in an editor.
- **File Uploads & Input Handling**: Users can upload files and enter input data, which will be passed to the executed code.
- **Execution Environment**: The generated code is executed in a local Python environment, and the results (output or errors) are displayed in real-time.

 

### Technologies Used:
- **Streamlit**: For creating the interactive web interface.
- **Ollama**: For integrating the Llama2 model for code generation.
- **Python**: For executing the generated code.
- **Subprocess**: To run the Python code and capture the output.

## Installation

### 1. Install **Ollama**
   - Download and install Ollama from [here](https://ollama.com).
   - Open the Ollama command prompt and run the following command to start the Llama2 model:
     ```bash
     ollama run llama2
     ```

### 2. Install **Anaconda**
   - Download and install Anaconda from [here](https://www.anaconda.com/products/distribution).
   - After installation, create a new Conda environment using the provided `AdvancedProgramming.yml` file. You can create an environment by running the following command in your terminal:
     ```bash
     conda env create -f AdvancedProgramming.yml
     ```
   - Once the environment is created, activate the environment using:
     ```bash
     conda activate <environment-name>
     ```

### 3. Run the Application
   - After all dependencies are installed, run the application using:
     ```bash
     streamlit run app.py
     ```

 

---

Feel free to reach out with any issues or for further assistance.
