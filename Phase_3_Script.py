import ollama
import json
import logging
import sys
import time
import threading
from typing import Dict, List
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from tqdm import tqdm  # Importing tqdm for progress bars
import ast
import astor  # You might need to install astor using pip

# ==========================
# Configuration Section
# ==========================

# Set the path to the folder containing summary files
SUMMARY_FOLDER_PATH = '/mmfs1/scratch/jacks.local/hdubey/Ollama/summaries'  # <-- Set your summary folder path here

# Set the path to the folder where scripts and results will be saved
OUTPUT_FOLDER_PATH = '/mmfs1/scratch/jacks.local/hdubey/Ollama/feasibility'  # <-- Set your output folder path here

# Path to the Z3 API mapping JSON file
Z3_API_MAPPING_PATH = '/mmfs1/scratch/jacks.local/hdubey/Ollama/z3.json'  # <-- Update this path accordingly

# Number of parallel threads for agent execution
MAX_WORKERS = 6  # Adjust based on your system's capabilities

# Maximum number of retry attempts for fixing scripts
MAX_RETRIES = 6

# ==========================
# Setup Logging
# ==========================
logging.basicConfig(
    filename='agent_workflow.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# ==========================
# Context Class
# ==========================
class Context:
    def __init__(self, summary_file: str, data_flow_paths: List[Dict[str, str]]):
        self.summary_file = summary_file
        self.data_flow_paths = data_flow_paths  # List of dictionaries with path details
        self.z3_scripts: Dict[str, str] = {}    # path_id -> script content
        self.results: Dict[str, Dict[str, str]] = {}  # path_id -> {'status': ..., 'error': ...}
        self.summary_content: str = ""  # To store the raw summary content

    def to_dict(self):
        return {
            "summary_file": self.summary_file,
            "data_flow_paths": self.data_flow_paths,
            "z3_scripts": self.z3_scripts,
            "results": self.results
        }

# ==========================
# Spinner Class
# ==========================
class Spinner:
    def __init__(self, message="Processing"):
        self.spinner_cycle = ['|', '/', '-', '\\']
        self.message = message
        self.running = False
        self.thread = None

    def spinner_task(self):
        idx = 0
        while self.running:
            sys.stdout.write(f"\r{self.message} {self.spinner_cycle[idx % len(self.spinner_cycle)]}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')  # Clear the line

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.spinner_task)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

# ==========================
# Agent Creation
# ==========================
def create_agent(step_number: int, step_description: str, output_format: str):
    """
    Creates an agent function that communicates with the Ollama model to perform specific tasks.
    
    Args:
        step_number (int): The step number for logging purposes.
        step_description (str): Description of the task the agent should perform.
        output_format (str): The expected output format from the agent.
    
    Returns:
        function: A function that takes a task input and returns the agent's response.
    """
    def agent(task: str):
        logger.info(f"Agent {step_number} - Task: {step_description}")
        logger.info(f"Agent {step_number} - Received Task Input: {task}")
        
        response = ollama.chat(
            model='nemotron',
            messages=[
                {
                    'role': 'system',
                    'content': (
                        f'You are Agent {step_number}. Your task is to {step_description}. '
                        f'Provide the output strictly in the following format:\n{output_format}'
                    )
                },
                {
                    'role': 'user',
                    'content': f'Given this task: {task}\nExecute step {step_number} as described.'
                }
            ]
        )
        result = response['message']['content']
        logger.info(f"Agent {step_number} - Response: {result}")
        return result
    return agent

# ==========================
# CEO Agent Function
# ==========================
def CEO_Agent(context: Context, mode: str = 'generate_z3_script', current_path: Dict[str, str] = None, agent_output: str = None, z3_api_mapping: Dict[str, str] = {}) -> str:
    """
    Central orchestrator agent handling various modes of operation.
    
    Args:
        context (Context): The current processing context.
        mode (str): The mode of operation for the CEO Agent.
        current_path (Dict[str, str], optional): The current data flow path being processed.
        agent_output (str, optional): The output from a previous agent task (e.g., error messages).
        z3_api_mapping (Dict[str, str], optional): The Z3 API mapping dictionary.
    
    Returns:
        str: The CEO Agent's response based on the mode.
    """
    if mode == 'extract_data_flow_paths':
        system_prompt = (
            "You are the CEO Agent. Your task is to extract all data flow paths from the following summary. "
            "Provide the extracted paths in a structured JSON format with each path containing 'Path ID', 'Source', 'Sink', and 'Security Note'. "
            "Ensure the JSON is valid and properly formatted. Do not include any additional text or markdown."
        )
        user_content = context.summary_content
    elif mode == 'generate_z3_script':
        system_prompt = (
            "You are the CEO Agent. Your task is to generate a Python script using the Z3 solver library to test the feasibility of the following data flow path. "
            "Provide only the Python code without any explanations or comments. Ensure the script is syntactically correct and uses appropriate Z3 functions (e.g., UDiv for unsigned division). "
            "Do not include any Markdown syntax or code block delimiters."
        )
        user_content = f"""Data Flow Path:
Source: {current_path['Source']}
Sink: {current_path['Sink']}
Security Note: {current_path.get('Security Note', 'None')}
"""
    elif mode == 'fix_script':
        # Enhanced fix_script mode
        # Step 1: Identify the problematic API from the error message
        problematic_api = identify_problematic_api(agent_output, z3_api_mapping)
        if problematic_api:
            # Step 2: Retrieve the description of the problematic API from the mapping
            api_description = z3_api_mapping.get(problematic_api, "No description available.")
            
            # Step 3: Formulate the prompt including the mapping information
            system_prompt = (
                "You are the CEO Agent. Based on the following error message, the original Python Z3 script, and the Z3 API mapping provided, "
                "identify and correct the issue related to Z3 API usage. "
                "Provide only the corrected Python code without any explanations or comments."
            )
            user_content = f"""Error Message:
{agent_output}

Original Script:
{current_path['Script']}

Z3 API Mapping:
{json.dumps(z3_api_mapping, indent=2)}
"""
        else:
            # If no problematic API is identified, provide general instructions
            system_prompt = (
                "You are the CEO Agent. Based on the following error message and the original Python Z3 script, fix the script to resolve the issue. "
                "Ensure that the corrected script uses appropriate Z3 functions and adheres to proper Python syntax. "
                "Do not include any Markdown syntax or explanations. Provide only the corrected Python code."
            )
            user_content = f"""Error Message:
{agent_output}

Original Script:
{current_path['Script']}
"""
    elif mode == 'summarize_results':
        system_prompt = (
            "You are the CEO Agent. Compile the feasibility results of all data flow paths into a concise summary. "
            "Indicate whether each path is feasible or not based on the script execution results."
        )
        # Compile user_content from context.results
        summary = ""
        for path_id, result in context.results.items():
            summary += f"Path {path_id}: {'FEASIBLE' if result['status'] == 'Success' else 'NOT FEASIBLE'}\n"
        user_content = summary
    else:
        raise ValueError("Invalid mode for CEO_Agent.")
    
    logger.info(f"CEO Agent - Mode: {mode}")
    if mode != 'summarize_results':
        logger.info(f"CEO Agent - User Content: {user_content}")
    
    response = ollama.chat(
        model='nemotron',
        messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_content
            }
        ]
    )
    result = response['message']['content']
    
    if mode == 'extract_data_flow_paths':
        logger.info(f"CEO Agent extracted Data Flow Paths:\n{result}")
    elif mode == 'generate_z3_script':
        logger.info(f"CEO Agent generated Z3 Script:\n{result}")
    elif mode == 'fix_script':
        logger.info(f"CEO Agent fixed Z3 Script:\n{result}")
    elif mode == 'summarize_results':
        logger.info(f"CEO Agent compiled feasibility summary:\n{result}")
    
    return result

# ==========================
# Helper Functions
# ==========================
def read_summary_file(file_path: str) -> str:
    """
    Reads the content of a summary file.

    Args:
        file_path (str): Path to the summary file.

    Returns:
        str: The content of the summary file.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        logger.info(f"Successfully read summary file: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading summary file {file_path}: {e}")
        return ""

def load_z3_api_mapping(json_path: str) -> Dict[str, str]:
    """
    Loads the Z3 API mapping from a JSON file.
    
    Args:
        json_path (str): The file path to the JSON mapping.
        
    Returns:
        Dict[str, str]: A dictionary mapping Z3 APIs to their descriptions.
    """
    try:
        with open(json_path, 'r') as file:
            z3_api_mapping = json.load(file)
        logger.info(f"Successfully loaded Z3 API mapping from {json_path}")
        return z3_api_mapping
    except Exception as e:
        logger.error(f"Failed to load Z3 API mapping from {json_path}: {e}")
        return {}

def identify_problematic_api(error_message: str, z3_api_mapping: Dict[str, str]) -> str:
    """
    Identifies the problematic Z3 API from the error message using the Z3 API mapping.
    
    Args:
        error_message (str): The error message from script execution.
        z3_api_mapping (Dict[str, str]): The Z3 API mapping dictionary.
        
    Returns:
        str: The identified problematic Z3 API or an empty string if not found.
    """
    # Iterate through the API mapping to find a matching API in the error message
    for api in z3_api_mapping.keys():
        # Exact match
        if api in error_message:
            logger.info(f"Identified problematic API: {api}")
            return api
        # Check for function calls (e.g., IsNonZero(), Equal())
        api_pattern = re.escape(api) + r'\s*\('
        if re.search(api_pattern, error_message):
            logger.info(f"Identified problematic API via pattern: {api}")
            return api
    # If no API matches, attempt to extract function names from the error message
    pattern = re.compile(r"NameError: name '(\w+)' is not defined")
    match = pattern.search(error_message)
    if match:
        undefined_api = match.group(1)
        if undefined_api in z3_api_mapping:
            logger.info(f"Identified undefined API: {undefined_api}")
            return undefined_api
    return ""

def extract_json_from_response(response: str) -> str:
    """
    Extracts JSON content from the CEO Agent's response.
    Assumes the JSON is enclosed within ```json and ```
    
    Args:
        response (str): The CEO Agent's response.
    
    Returns:
        str: The extracted JSON string.
    """
    json_start = response.find('```json')
    json_end = response.find('```', json_start + 7)
    if json_start != -1 and json_end != -1:
        json_str = response[json_start + 7:json_end].strip()
        return json_str
    else:
        # Attempt to find JSON without markdown
        try:
            json_str = re.search(r'({.*})', response, re.DOTALL).group(1)
            return json_str
        except AttributeError:
            return ''

def extract_code_from_response(response: str) -> str:
    """
    Extracts Python code from the CEO Agent's response.
    Removes any Markdown code block delimiters.
    
    Args:
        response (str): The CEO Agent's response.
    
    Returns:
        str: The extracted Python code.
    """
    code_block_pattern = re.compile(r'```python\s*\n(.*?)\n```', re.DOTALL)
    match = code_block_pattern.search(response)
    if match:
        return match.group(1).strip()
    else:
        # If no markdown delimiters are found, return the whole response
        return response.strip()

def validate_script_syntax(script_content: str) -> bool:
    """
    Validates the syntax of the given Python script.
    
    Args:
        script_content (str): The Python script content.
    
    Returns:
        bool: True if syntax is valid, False otherwise.
    """
    try:
        ast.parse(script_content)
        logger.info("Script syntax is valid.")
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in script: {e}")
        return False

class Z3APICorrectionTransformer(ast.NodeTransformer):
    def __init__(self, problematic_api: str, z3_api_mapping: Dict[str, str]):
        self.problematic_api = problematic_api
        self.z3_api_mapping = z3_api_mapping

    def visit_Call(self, node):
        # Check if the function being called is the problematic API
        if isinstance(node.func, ast.Name) and node.func.id == self.problematic_api:
            # Example: Replace IsNonZero(divisor) with (divisor != 0)
            # This logic can be customized based on specific API corrections
            if self.problematic_api == 'IsNonZero':
                # Assuming one argument
                if len(node.args) == 1:
                    arg = node.args[0]
                    new_node = ast.Compare(
                        left=arg,
                        ops=[ast.NotEq()],
                        comparators=[ast.Num(n=0)]
                    )
                    return ast.copy_location(new_node, node)
            # Add more conditions for different APIs as needed
        return self.generic_visit(node)

def apply_ast_corrections(script_content: str, problematic_api: str, z3_api_mapping: Dict[str, str]) -> str:
    """
    Applies AST-based corrections to the script based on the problematic API.
    
    Args:
        script_content (str): The Python script content.
        problematic_api (str): The API that needs correction.
        z3_api_mapping (Dict[str, str]): The Z3 API mapping dictionary.
    
    Returns:
        str: The corrected Python script content.
    """
    tree = ast.parse(script_content)
    transformer = Z3APICorrectionTransformer(problematic_api, z3_api_mapping)
    corrected_tree = transformer.visit(tree)
    corrected_script = astor.to_source(corrected_tree)
    return corrected_script

def validate_and_correct_code(script_content: str, z3_api_mapping: Dict[str, str]) -> str:
    """
    Validates the extracted Python script for known Z3 API issues and corrects them.
    
    Args:
        script_content (str): The Python script content.
        z3_api_mapping (Dict[str, str]): The Z3 API mapping dictionary.
    
    Returns:
        str: The corrected Python script content.
    """
    # Initial regex-based corrections (if any)
    # Example: Replace 'z3.Div' with 'UDiv' for unsigned integer division
    script_content = re.sub(r'\bz3\.Div\b', 'UDiv', script_content)
    
    # Replace 'Len' with 'Length' for string length calculations in Z3
    script_content = re.sub(r'\bLen\b', 'Length', script_content)
    
    # Replace 'fabs' with 'Abs' for absolute value in Z3
    script_content = re.sub(r'\bfabs\b', 'Abs', script_content)
    
    # Correct improper usage of assert_and_track
    # Replace list inputs with individual expressions
    script_content = re.sub(
        r's\.assert_and_track\(\s*\[(.*?)\],\s*[\'"](.*?)[\'"]\s*\)',
        r's.assert_and_track(\1, "\2")',
        script_content
    )
    
    # Replace 'z3.sat' with 'sat' and 'z3.unsat' with 'unsat' if necessary
    script_content = re.sub(r'\bz3\.sat\b', 'sat', script_content)
    script_content = re.sub(r'\bz3\.unsat\b', 'unsat', script_content)
    
    # Ensure 'replace' function is imported if used
    if 'replace(' in script_content and 'from z3 import replace' not in script_content:
        script_content = 'from z3 import replace\n' + script_content
    
    return script_content

def extract_data_flow_paths_via_CEO(context: Context, z3_api_mapping: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Uses the CEO Agent to extract data flow paths from summary content.
    
    Args:
        context (Context): The current processing context.
        z3_api_mapping (Dict[str, str]): The Z3 API mapping dictionary.
    
    Returns:
        List[Dict[str, str]]: A list of extracted data flow paths.
    """
    spinner = Spinner("Extracting Data Flow Paths")
    spinner.start()
    extracted_response = CEO_Agent(context, mode='extract_data_flow_paths', z3_api_mapping=z3_api_mapping)
    spinner.stop()
    
    extracted_json = extract_json_from_response(extracted_response)
    
    if not extracted_json:
        logger.error("Failed to extract JSON from CEO Agent's response.")
        logger.error(f"CEO Agent's Response:\n{extracted_response}")
        return []
    
    try:
        data_flow_paths_json = json.loads(extracted_json)
        data_flow_paths = data_flow_paths_json.get("Data Flow Paths", [])
        
        # Normalize Path IDs to include 'DFP-' prefix if not already present
        for path in data_flow_paths:
            if not path['Path ID'].startswith('DFP-'):
                # Assuming Path ID is like 'PATH1', 'PATH2', etc.
                match = re.match(r'PATH(\d+)', path['Path ID'])
                if match:
                    numeric_id = match.group(1).zfill(3)
                    path['Path ID'] = f"DFP-{numeric_id}"
                else:
                    # If Path ID does not match expected pattern, retain as is or handle accordingly
                    path['Path ID'] = f"DFP-{path['Path ID']}"
        
        logger.info(f"Extracted {len(data_flow_paths)} data flow paths via CEO Agent.")
        return data_flow_paths
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from CEO Agent's response: {e}")
        logger.error(f"Extracted JSON:\n{extracted_json}")
        return []

def execute_z3_script(script_content: str, script_path: str) -> Dict[str, str]:
    """
    Executes the Z3 script and returns the result.
    
    Args:
        script_content (str): The Python script content.
        script_path (str): The file path where the script is saved.
    
    Returns:
        Dict[str, str]: A dictionary containing 'status' and 'error' (if any).
    """
    try:
        # Write the script to a temporary file
        with open(script_path, 'w') as script_file:
            script_file.write(script_content)
        logger.info(f"Executing script: {script_path}")
        
        # Run the script using subprocess
        result = subprocess.run(['python3', script_path], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info(f"Script executed successfully: {script_path}")
            logger.info(f"Script Output:\n{result.stdout}")
            return {'status': 'Success', 'error': ''}
        else:
            logger.error(f"Script execution failed: {script_path}\nError: {result.stderr}")
            return {'status': 'Error', 'error': result.stderr}
    except subprocess.TimeoutExpired:
        logger.error(f"Script execution timed out: {script_path}")
        return {'status': 'Timeout', 'error': 'Execution timed out.'}
    except Exception as e:
        logger.error(f"Unexpected error during script execution: {script_path}\nError: {e}")
        return {'status': 'Exception', 'error': str(e)}

def save_script_and_result(output_dir: str, path_id: str, script_content: str, result: Dict[str, str]):
    """
    Saves the Z3 script and its execution result to the specified directory.
    
    Args:
        output_dir (str): Directory where the script and result will be saved.
        path_id (str): The ID of the data flow path.
        script_content (str): The Python script content.
        result (Dict[str, str]): The result of script execution.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        script_filename = f"z3_script_path_{path_id}.py"
        script_path = os.path.join(output_dir, script_filename)
        
        # Save the script
        with open(script_path, 'w') as script_file:
            script_file.write(script_content)
        logger.info(f"Saved Z3 script: {script_path}")
        
        # Save the result
        result_filename = f"result_path_{path_id}.txt"
        result_path = os.path.join(output_dir, result_filename)
        with open(result_path, 'w') as result_file:
            result_file.write(f"Execution Status: {result['status']}\n")
            if result['error']:
                result_file.write(f"Error Message:\n{result['error']}\n")
            else:
                # Optionally, save the standard output
                pass
        logger.info(f"Saved execution result: {result_path}")
    except Exception as e:
        logger.error(f"Failed to save script and result for Path {path_id}: {e}")

def handle_path(path: Dict[str, str], output_dir: str, context: Context, z3_api_mapping: Dict[str, str]):
    """
    Handles the processing of a single data flow path: generates, executes, and fixes scripts as needed.
    
    Args:
        path (Dict[str, str]): The data flow path details.
        output_dir (str): Directory where outputs will be saved.
        context (Context): The current processing context.
        z3_api_mapping (Dict[str, str]): The Z3 API mapping dictionary.
    """
    path_id = path['Path ID']
    source = path['Source']
    sink = path['Sink']
    security_note = path.get('Security Note', 'None')
    
    # Extract numerical part of Path ID for step_number
    # Expected Path ID format: DFP-001, DFP-002, etc.
    match = re.match(r'DFP-(\d+)', path_id)
    if match:
        step_number = int(match.group(1))
    else:
        # Attempt to extract numeric part from 'PATH1', 'PATH2', etc.
        match = re.search(r'\d+', path_id)
        if match:
            step_number = int(match.group())
        else:
            logger.error(f"Invalid Path ID format: {path_id}")
            return
    
    # Step 2: Generate Z3 Script
    agent_description = (
        f"Generate a Python script using the Z3 solver library to test the feasibility of the data flow path "
        f"from {source} to {sink}. Ensure the script is syntactically correct, uses appropriate Z3 functions (e.g., UDiv for unsigned division), "
        f"and does not include any Markdown syntax. Provide only the Python code without any explanations or comments."
    )
    output_format = "Provide only the Python script without any additional explanations or markdown."
    agent = create_agent(step_number=step_number, step_description=agent_description, output_format=output_format)
    
    logger.info(f"Generating Z3 Script for Path {path_id}")
    script_response = agent(f"Data Flow Path ID: {path_id}")
    
    # Extract code from response
    script_content = extract_code_from_response(script_response)
    if not script_content:
        logger.error(f"Failed to extract Python code for Path {path_id}")
        return
    
    # Validate and correct the code
    script_content = validate_and_correct_code(script_content, z3_api_mapping)
    
    context.z3_scripts[path_id] = script_content
    
    # Initialize retry mechanism
    attempt = 0
    success = False
    
    while attempt < MAX_RETRIES and not success:
        attempt += 1
        logger.info(f"Executing script (Attempt {attempt}) for Path {path_id}")
        script_path = os.path.join(output_dir, f"z3_script_path_{path_id}.py")
        result = execute_z3_script(script_content, script_path)
        context.results[path_id] = result
        
        if result['status'] == 'Success':
            logger.info(f"Script executed successfully on attempt {attempt} for Path {path_id}")
            success = True
        else:
            logger.warning(f"Script execution failed on attempt {attempt} for Path {path_id}. Error: {result['error']}")
            if attempt < MAX_RETRIES:
                logger.info(f"Attempting to fix script for Path {path_id}")
                # Add the script content to current_path for fixing
                path['Script'] = script_content
                fix_response = CEO_Agent(context, mode='fix_script', current_path=path, agent_output=result['error'], z3_api_mapping=z3_api_mapping)
                
                # Extract fixed code from response
                fixed_script = extract_code_from_response(fix_response)
                if not fixed_script:
                    logger.error(f"Failed to extract fixed Python code for Path {path_id}")
                    break  # Exit the loop if fixing failed
                
                # Validate and correct the fixed code
                fixed_script = validate_and_correct_code(fixed_script, z3_api_mapping)
                
                # Apply AST corrections based on the problematic API
                problematic_api = identify_problematic_api(result['error'], z3_api_mapping)
                if problematic_api:
                    fixed_script = apply_ast_corrections(fixed_script, problematic_api, z3_api_mapping)
                
                # Validate script syntax before proceeding
                if validate_script_syntax(fixed_script):
                    script_content = fixed_script  # Update the script content for the next attempt
                    context.z3_scripts[path_id] = script_content
                else:
                    logger.error(f"Syntax validation failed after corrections for Path {path_id}.")
                    break
            else:
                logger.error(f"Exceeded maximum retries for Path {path_id}. Moving to the next path.")
    
    # Step 5: Save Script and Result if successful
    if success:
        save_script_and_result(output_dir, path_id, context.z3_scripts[path_id], context.results[path_id])
    else:
        # Save the last failed script and error for further analysis
        save_script_and_result(output_dir, path_id, context.z3_scripts[path_id], context.results[path_id])

# ==========================
# Main Workflow Function
# ==========================
def process_summary(summary_file_path: str, output_base_path: str, z3_api_mapping: Dict[str, str]):
    """
    Processes a single summary file: extracts DFPs, generates and executes scripts, handles errors, and compiles results.
    
    Args:
        summary_file_path (str): Path to the summary file.
        output_base_path (str): Base directory where outputs will be saved.
        z3_api_mapping (Dict[str, str]): The Z3 API mapping dictionary.
    """
    summary_content = read_summary_file(summary_file_path)
    if not summary_content:
        logger.error(f"Empty or failed to read summary file: {summary_file_path}")
        return
    
    summary_filename = os.path.basename(summary_file_path)
    summary_name, _ = os.path.splitext(summary_filename)
    output_dir = os.path.join(output_base_path, summary_name)
    
    # Ensure the output directory is created successfully
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return
    
    context = Context(summary_file=summary_filename, data_flow_paths=[])
    context.summary_content = summary_content  # Add summary_content to context for CEO Agent
    
    logger.info(f"Processing summary file: {summary_filename}")
    
    # Step 1: Extract Data Flow Paths via CEO Agent
    data_flow_paths = extract_data_flow_paths_via_CEO(context, z3_api_mapping)
    if not data_flow_paths:
        logger.warning(f"No data flow paths extracted for summary file: {summary_filename}")
        return
    context.data_flow_paths = data_flow_paths
    
    # Use ThreadPoolExecutor to handle multiple paths in parallel with tqdm progress bar
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(handle_path, path, output_dir, context, z3_api_mapping) for path in data_flow_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Paths", unit="path"):
            try:
                future.result()
                # No need to log here as logging is handled within handle_path
            except Exception as e:
                # Identify which path caused the exception
                # Using exception traceback to find the path_id
                logger.error(f"Error processing a path: {e}")

    # Step 6: Summarize Results
    spinner = Spinner("Summarizing Feasibility Results")
    spinner.start()
    summary = CEO_Agent(context, mode='summarize_results', z3_api_mapping=z3_api_mapping)
    spinner.stop()
    
    # Save the summary
    summary_filename = os.path.join(output_dir, 'feasibility_summary.txt')
    try:
        with open(summary_filename, 'w') as summary_file:
            summary_file.write(summary)
        logger.info(f"Saved feasibility summary: {summary_filename}")
    except Exception as e:
        logger.error(f"Failed to save feasibility summary: {summary_filename}\nError: {e}")

# ==========================
# Execution Block
# ==========================
if __name__ == "__main__":
    # Load the Z3 API mapping
    z3_api_mapping = load_z3_api_mapping(Z3_API_MAPPING_PATH)
    
    # Check if the summary folder exists
    if not os.path.isdir(SUMMARY_FOLDER_PATH):
        logger.error(f"Summary folder does not exist: {SUMMARY_FOLDER_PATH}")
        print(f"Summary folder does not exist: {SUMMARY_FOLDER_PATH}")
        sys.exit(1)
    
    # Ensure the output folder exists; create it if it doesn't
    try:
        os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
        logger.info(f"Output folder is set to: {OUTPUT_FOLDER_PATH}")
    except Exception as e:
        logger.error(f"Failed to create or access output folder {OUTPUT_FOLDER_PATH}: {e}")
        print(f"Failed to create or access output folder: {OUTPUT_FOLDER_PATH}")
        sys.exit(1)
    
    # Traverse the directory recursively and list all summary files (assuming .txt extension)
    summary_files = []
    for root, dirs, files in os.walk(SUMMARY_FOLDER_PATH):
        for file in files:
            if file.endswith('.txt'):
                summary_files.append(os.path.join(root, file))
    
    if not summary_files:
        logger.warning(f"No summary files found in: {SUMMARY_FOLDER_PATH}")
        print(f"No summary files found in: {SUMMARY_FOLDER_PATH}")
        sys.exit(0)
    
    logger.info(f"Found {len(summary_files)} summary files to process.")
    
    # Process each summary file
    for summary_file in summary_files:
        print(f"\nProcessing summary file: {os.path.basename(summary_file)}")
        logger.info(f"Starting processing for summary file: {summary_file}")
        process_summary(summary_file, OUTPUT_FOLDER_PATH, z3_api_mapping)
        print(f"Completed processing for summary file: {os.path.basename(summary_file)}")
