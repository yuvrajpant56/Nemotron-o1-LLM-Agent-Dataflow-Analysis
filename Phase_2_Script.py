import ollama
import re
import json
import logging
import sys
import time
import threading
from typing import Dict, List
import os

# ==========================
# Configuration Section
# ==========================

# Set the path to the folder containing .c or .cpp files
INPUT_FOLDER_PATH =  <-- Set your input folder path here

# Set the path to the folder where summaries will be saved
OUTPUT_FOLDER_PATH = <-- Set your output folder path here

# ==========================
# Setup Logging
# ==========================
logging.basicConfig(
    filename='workflow.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# ==========================
# Context Class
# ==========================
class Context:
    def __init__(self, original_input: str, overall_goal: str):
        self.original_input = original_input
        self.overall_goal = overall_goal
        self.initial_cot = ""
        self.refined_cot = ""
        self.steps: List[str] = []
        self.implementations: Dict[str, str] = {}
        self.final_summary = ""

    def to_dict(self):
        return {
            "original_input": self.original_input,
            "overall_goal": self.overall_goal,
            "initial_cot": self.initial_cot,
            "refined_cot": self.refined_cot,
            "steps": self.steps,
            "implementations": self.implementations,
            "final_summary": self.final_summary
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
# CEO Agent
# ==========================
def CEO_Agent(context: Context, mode: str = 'generate_cot', current_step: str = None, agent_output: str = None):
    """
    Modes:
    - 'generate_cot': Generate initial Chain of Thought (CoT)
    - 'reflect_cot': Refine the initial CoT
    - 'verify_agent_output': Verify the output of a specialized agent
    - 'summarize': Compile the final summary
    """
    if mode == 'generate_cot':
        system_prompt = (
            "You are the CEO Agent. Analyze the following task and generate a concise, dependent Chain of Thought (CoT) "
            "consisting of clear and necessary steps to extract data flow paths from sources to sinks in a divide and conquer manner. "
            "Ensure that the output of one step can be utilized by the subsequent step. "
            "Determine all intermediate outputs needed to generate the final data flow path.\n"
            "Use as few steps as necessary to effectively address the problem.\n"
            "Format each step precisely as follows, ensuring double asterisks and correct numbering without any additional text:\n"
            "**1. ** **Step Title**\n"
            "    * Description of the step."
        )
        user_content = context.overall_goal
    elif mode == 'reflect_cot':
        system_prompt = (
            "You are the CEO Agent. Review the following Chain of Thought (CoT) and refine it to make it "
            "as concise as possible while maintaining dependencies between steps. Remove any redundant or unnecessary steps, "
            "combine steps where logical, and ensure there are no more than 2 steps. The refined CoT should strictly focus on extracting data flow paths from sources to sinks in a divide and conquer manner, without adding any extra analysis or unrelated information.\n"
            "Maintain the following precise format for each step, ensuring double asterisks and correct numbering:\n"
            "**1. ** **Step Title**\n"
            "    * Description of the step."
        )
        user_content = context.initial_cot
    elif mode == 'verify_agent_output':
        system_prompt = (
            "You are the CEO Agent. Review the following agent response and verify its correctness and adherence to the expected format.\n"
            f"Overall Goal: {context.overall_goal}\n"
            f"Current Step: {current_step}\n"
            f"Agent Output:\n{agent_output}\n"
            "Provide feedback on whether the response is correct and properly formatted. If not, suggest necessary corrections."
        )
        user_content = ""
    elif mode == 'summarize':
        system_prompt = (
            "You are the CEO Agent. Summarize the implementations provided for each step into a final comprehensive summary "
            "focusing solely on data flow paths from sources to sinks. Ensure that each data flow path includes the source declaration/definition, the code lines where the source is used, and the full sink code. Do not include any additional notes, recommendations, or unrelated information."
        )
        # Prepare the input by compiling steps and their implementations
        cot_steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(context.steps)])
        implementations = "\n".join([
            f"**Step {i+1}:** {impl}" for i, impl in enumerate(context.implementations.values())
        ])
        user_content = f"Chain of Thought Steps:\n{cot_steps}\n\nImplementations:\n{implementations}"
    else:
        raise ValueError("Invalid mode for CEO_Agent.")
    
    logger.info(f"CEO Agent - Mode: {mode}")
    if mode != 'verify_agent_output':
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
    
    if mode == 'generate_cot':
        context.initial_cot = result
        logger.info(f"CEO Agent generated initial Chain of Thought (CoT):\n{result}")
    elif mode == 'reflect_cot':
        context.refined_cot = result
        logger.info(f"CEO Agent reflected and refined Chain of Thought (CoT):\n{result}")
    elif mode == 'verify_agent_output':
        logger.info(f"CEO Agent verification feedback for Step '{current_step}':\n{result}")
    elif mode == 'summarize':
        context.final_summary = result
        logger.info(f"CEO Agent final summary:\n{result}")
    return result

# ==========================
# Helper Functions
# ==========================
def preprocess_cot(cot: str) -> str:
    """
    Preprocesses the Chain of Thought (CoT) to ensure consistent formatting.
    Ensures all step titles have opening and closing **.
    """
    cot = re.sub(r'^(\*\*\d+\.\s*\*\*)([^*]+?)(\*\*)$', r'\1**\2**', cot, flags=re.MULTILINE)
    logger.info(f"Preprocessed CoT:\n{cot}")
    return cot

def determine_output_format(step_description: str) -> str:
    """
    Determines the output format based on keywords in the step description.
    """
    # Define keyword-based mappings
    format_mappings = {
        'identify': "List of sources and sinks.",
        'refine': "List of sources and sinks with explanations.",
        'map': "List of data flows with their locations in the code (line numbers).",
        'construct': "Detailed data flow paths from each source to each sink, including source declaration/definition, the code lines where the source is used, and the full sink code."
    }
    
    # Lowercase step description for matching
    step_desc_lower = step_description.lower()
    
    for keyword, fmt in format_mappings.items():
        if keyword in step_desc_lower:
            logger.info(f"Determined output format for step '{step_description}' as: {fmt}")
            return fmt
    
    # Default format if no keyword matches
    logger.info(f"No keyword matched for step '{step_description}'. Using default format.")
    return "Provide the results in a clear and structured format."

def extract_relevant_info(context: Context, step_number: int) -> str:
    """
    Extracts only the necessary information from previous agent outputs to pass to the next agent.
    """
    if step_number == 2:
        # Extract sources and sinks from Step 1
        step1_output = context.implementations.get("Step 1", "")
        # Use regex to find sources and sinks
        sources = re.findall(r'Source[s]?:\s*(.*)', step1_output, re.IGNORECASE)
        sinks = re.findall(r'Sink[s]?:\s*(.*)', step1_output, re.IGNORECASE)
        extracted_info = ""
        if sources:
            extracted_info += f"Sources: {sources[0].strip()}\n"
        if sinks:
            extracted_info += f"Sinks: {sinks[0].strip()}\n"
        return extracted_info.strip()
    elif step_number == 3:
        # Extract data flows from Step 2
        step2_output = context.implementations.get("Step 2", "")
        # Assume data flows are listed in some recognizable pattern
        data_flows = re.findall(r'Data Flows?:\s*(.*)', step2_output, re.IGNORECASE)
        extracted_info = ""
        if data_flows:
            extracted_info += f"Data Flows: {data_flows[0].strip()}\n"
        return extracted_info.strip()
    elif step_number == 4:
        # Extract vulnerability analysis from Step 3
        step3_output = context.implementations.get("Step 3", "")
        # Assume vulnerabilities are listed in some recognizable pattern
        vulnerabilities = re.findall(r'Vulnerabilities?:\s*(.*)', step3_output, re.IGNORECASE)
        extracted_info = ""
        if vulnerabilities:
            extracted_info += f"Vulnerabilities: {vulnerabilities[0].strip()}\n"
        return extracted_info.strip()
    else:
        return context.original_input  # For Step 1, pass the original input

def verify_and_fix_response(context: Context, step_number: int, step_description: str, agent_output: str):
    """
    Verifies the agent's output and attempts to fix it if necessary.
    """
    # Determine expected output format dynamically based on step description
    output_format = determine_output_format(step_description)
    
    # CEO Agent verifies the agent's output
    verification_feedback = CEO_Agent(
        context,
        mode='verify_agent_output',
        current_step=step_description,
        agent_output=agent_output
    )
    
    # Determine if the response is correct based on the feedback
    if any(keyword in verification_feedback.lower() for keyword in ['yes', 'correct', 'properly formatted']):
        logger.info(f"Agent {step_number} output verified as correct.")
        return agent_output
    else:
        # Determine if the current step is "Identify Sources and Sinks" to set a specific spinner message
        if step_number == 1 and 'identify sources and sinks' in step_description.lower():
            spinner_message = "Fixing - Extracted source and sink"
        else:
            spinner_message = "Fixing"

        # Flash the specific spinner message
        spinner = Spinner(spinner_message)
        spinner.start()
        # CEO Agent suggests corrections; extract the suggested correction
        fix_prompt = (
            "You are the CEO Agent. Based on the verification feedback, correct the agent's response to ensure it meets the expected format and correctness."
        )
        correction_response = ollama.chat(
            model='nemotron',
            messages=[
                {
                    'role': 'system',
                    'content': fix_prompt
                },
                {
                    'role': 'user',
                    'content': f'Original Output:\n{agent_output}\nFeedback:\n{verification_feedback}\nProvide a corrected and properly formatted response.'
                }
            ]
        )
        corrected_output = correction_response['message']['content']
        spinner.stop()
        logger.info(f"Agent {step_number} output corrected to:\n{corrected_output}")
        return corrected_output

def extract_steps(cot: str) -> List[str]:
    """
    Extracts individual steps from the Chain of Thought (CoT).
    """
    # Updated regex to match steps formatted as **1. ** **Step Title**
    steps = re.findall(r'^\*\*\d+\.\s*\*\*\s*(.*?)\*\*\s*$', cot, re.MULTILINE)
    if not steps:
        # Attempt a more flexible regex
        steps = re.findall(r'^\*\*\d+\.\s*\*\*(.*?)\*\*\s*$', cot, re.MULTILINE)
    logger.info(f"Extracted steps: {steps}")
    return steps[:4]  # Ensure no more than 4 steps in the final CoT

# ==========================
# Main Workflow: process_task Function
# ==========================
def process_task(user_input: str) -> str:
    # Initialize context with original input and overall goal
    overall_goal = "Analyze the provided C/C++ code to identify divide-by-zero (DBZ) vulnerabilities by extracting data flow paths from sources to sinks."
    context = Context(original_input=user_input, overall_goal=overall_goal)
    
    logger.info("Process Task Initiated.")
    
    # Step 1: CEO generates initial CoT with spinner
    spinner = Spinner("Thinking")
    spinner.start()
    initial_cot = CEO_Agent(context, mode='generate_cot')
    spinner.stop()
    
    # Preprocess CoT to ensure consistent formatting
    context.initial_cot = preprocess_cot(context.initial_cot)
    
    # Step 2: CEO reflects on CoT to make it concise with spinner
    spinner = Spinner("Thinking")
    spinner.start()
    refined_cot = CEO_Agent(context, mode='reflect_cot')
    spinner.stop()
    
    # Preprocess refined CoT
    context.refined_cot = preprocess_cot(context.refined_cot)
    
    # Log the refined CoT for debugging
    logger.debug(f"Refined CoT:\n{context.refined_cot}")
    
    # Step 3: Extract steps from refined CoT (limit to 4 steps)
    spinner = Spinner("Thinking")
    spinner.start()
    context.steps = extract_steps(context.refined_cot)
    spinner.stop()
    
    # Flashing "Preliminary Data Flow Path Construction" after step extraction
    spinner = Spinner("Preliminary Data Flow Path Construction")
    spinner.start()
    time.sleep(2)  # Let the spinner flash for 2 seconds
    spinner.stop()
    
    if not context.steps:
        logger.error("No steps were extracted from the Chain of Thought. Please ensure the CoT is formatted correctly.")
        print("No steps were extracted from the Chain of Thought. Please ensure the CoT is formatted correctly.")
        return ""
    
    # Step 4: Create agents dynamically based on number of steps
    agents = []
    for i, step in enumerate(context.steps, start=1):
        output_format = determine_output_format(step)
        agents.append((i, step, output_format))
    
    # Step 5: Execute each agent sequentially, passing the necessary input
    for step_num, step_desc, out_format in agents:
        # Extract relevant input based on the current step
        if step_num == 1:
            relevant_input = context.original_input
        else:
            relevant_input = extract_relevant_info(context, step_num)
        
        # Start spinner with current step title
        spinner = Spinner(step_desc)
        spinner.start()
        
        # Execute the step via agent, passing the relevant_input
        agent_response = create_agent(step_num, step_desc, out_format)(relevant_input)
        
        # Verify and fix the response via CEO
        verified_response = verify_and_fix_response(context, step_num, step_desc, agent_response)
        
        # Store the implementation
        context.implementations[f"Step {step_num}"] = verified_response
        logger.info(f"Step {step_num} Implementation Stored.")
        
        # Stop the spinner after step execution
        spinner.stop()
    
    # Step 6: CEO summarizes the implementations (dataflow path summary only) with spinner
    spinner = Spinner("Summarizing")
    spinner.start()
    final_summary = CEO_Agent(context, mode='summarize')
    spinner.stop()
    context.final_summary = final_summary
    logger.info("Final Summary Compiled.")
    
    # Save final summary to file
    with open('final_summary.txt', 'w') as f:
        f.write(final_summary)
    logger.info("Final Summary Saved to 'final_summary.txt'.")
    
    return final_summary

# ==========================
# Function to Read C/C++ Files
# ==========================
def read_c_cpp_file(file_path: str) -> str:
    """
    Reads the content of a .c or .cpp file.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        logger.info(f"Successfully read file: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

# ==========================
# Example Usage
# ==========================
if __name__ == "__main__":
    # Check if the input folder exists
    if not os.path.isdir(INPUT_FOLDER_PATH):
        logger.error(f"Input folder does not exist: {INPUT_FOLDER_PATH}")
        print(f"Input folder does not exist: {INPUT_FOLDER_PATH}")
        sys.exit(1)
    
    # Ensure the output folder exists; create it if it doesn't
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
    logger.info(f"Output folder is set to: {OUTPUT_FOLDER_PATH}")
    
    # Traverse the directory recursively and list all .c and .cpp files
    for root, dirs, files in os.walk(INPUT_FOLDER_PATH):
        for file in files:
            if file.endswith(('.c', '.cpp')):
                file_path = os.path.join(root, file)
                print(f"\nProcessing file: {file}")
                logger.info(f"Processing file: {file_path}")
                
                # Read the file content
                code_content = read_c_cpp_file(file_path)
                if not code_content:
                    print(f"Failed to read file: {file}. Skipping.")
                    continue
                
                # Define sources and sinks based on comments or predefined information
                # You might need to adjust this part based on how sources and sinks are identified in C/C++ code
                # For demonstration, we'll assume sources and sinks are provided in comments at the end of the file
                
                # Example:
                # /*
                # Source: variable_x (derived from user input).
                # Sink: variable_y in function_do_something(int a, int b).
                # */
                
                # Extract sources and sinks from comments
                sources = re.findall(r'Source[s]?:\s*(.*)', code_content, re.IGNORECASE)
                sinks = re.findall(r'Sink[s]?:\s*(.*)', code_content, re.IGNORECASE)
                
                # Prepare the user input for processing
                user_input = f"""Analyze the following C/C++ code for divide-by-zero (DBZ) vulnerabilities. The code and identified sources and sinks are provided below. The goal is to find data flow path from sources to sinks. Code:
{code_content}
"""
                # Append sources and sinks if they were found
                if sources:
                    user_input += "Source: " + ", ".join(sources) + "\n"
                if sinks:
                    user_input += "Sink: " + ", ".join(sinks) + "\n"
                
                # Process the task
                final_result = process_task(user_input)
                print("Final Summary:\n", final_result)
                
                # ==========================
                # Save Final Summary to a Separate File in OUTPUT_FOLDER_PATH
                # ==========================
                # Extract the base filename without extension
                base_filename = os.path.splitext(file)[0]
                
                # Create a new filename for the data flow path summary
                data_flow_path_filename = f"data_flow_path_{base_filename}.txt"
                
                # Define the full path for the summary file in the output folder
                data_flow_path_fullpath = os.path.join(OUTPUT_FOLDER_PATH, data_flow_path_filename)
                
                # Save the final_summary (data flow path) to the separate file in OUTPUT_FOLDER_PATH
                try:
                    with open(data_flow_path_fullpath, 'w') as dfp_file:
                        dfp_file.write(final_result)
                    logger.info(f"Data Flow Path Summary Saved to '{data_flow_path_fullpath}'.")
                except Exception as e:
                    logger.error(f"Failed to save Data Flow Path Summary to '{data_flow_path_fullpath}': {e}")
                    print(f"Failed to save Data Flow Path Summary to '{data_flow_path_fullpath}'.")
