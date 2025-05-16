from camel.toolkits import (
    SearchToolkit,
    DocumentProcessingToolkit,
    FileWriteToolkit,
    FunctionTool
)
from camel.models import ModelFactory
from camel.types import(
    ModelPlatformType,
    ModelType
)

from camel.tasks import Task
from dotenv import load_dotenv

load_dotenv(override=True)

import os
import json
from typing import List, Dict, Any, Optional, Union, Literal
from loguru import logger
from utils import OwlWorkforce, OwlWorkforceChatAgent

import shutil


def construct_agent_list() -> List[Dict[str, Any]]:

    web_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict={"temperature": 0},
    )
    
    document_processing_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict={"temperature": 0},
    )

    search_toolkit = SearchToolkit()
    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp")
    file_write_toolkit = FileWriteToolkit("./tmp/")

    web_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that can search the web, extract webpage content, simulate browser actions, and provide relevant information to solve the given task.",
        model=web_model,
        tools=[
            FunctionTool(search_toolkit.search_duckduckgo),
            FunctionTool(document_processing_toolkit.extract_document_content),
        ]
    )
    
    document_processing_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that can process documents and multimodal data, such as images, audio, and video.",
        document_processing_model,
        tools=[
            FunctionTool(document_processing_toolkit.extract_document_content),
            FunctionTool(file_write_toolkit.write_to_file),
        ]
    )
    
    agent_list = []
    
    web_agent_dict = {
        "name": "Web Agent",
        "description": "A helpful assistant that can search the web, extract webpage content, simulate browser actions, and retrieve relevant information.",
        "agent": web_agent
    }
    
    document_processing_agent_dict = {
        "name": "Document Processing Agent",
        "description": "A helpful assistant that can process a variety of local and remote documents, including pdf, docx, images, audio, and video, etc.",
        "agent": document_processing_agent
    }

    agent_list.append(web_agent_dict)
    agent_list.append(document_processing_agent_dict)
    return agent_list


def construct_workforce() -> OwlWorkforce:
    """Construct a workforce with agents of different capabilities."""
    
    # Create models for the coordinator and task agents
    coordinator_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0},
        )
    }
    
    task_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0},
        )
    }
    
    # Create the workforce
    workforce = OwlWorkforce(
        "Simple Example Workforce",
        task_agent_kwargs=task_agent_kwargs,
        coordinator_agent_kwargs=coordinator_agent_kwargs,
    )

    # Add agents to the workforce
    agent_list = construct_agent_list()
    
    for agent_dict in agent_list:
        workforce.add_single_agent_worker(
            agent_dict["description"],
            worker=agent_dict["agent"],
            name=agent_dict["name"]
        )

    return workforce


def run_example_tasks():
    """Run example tasks using the workforce."""
    
    # Create the example tasks
    example_task = """
Make a simple travel plan for a 3-day trip to Paris, and save the plan to local file `travel_plan.md`.
"""
    
    # Clean temporary directories if they exist
    if os.path.exists("tmp/"):
        shutil.rmtree("tmp/")
    
    # Create the workforce
    workforce = construct_workforce()
    task = Task(content=example_task,)
    
    task_result = workforce.process_task(task)
    logger.success(task_result.result)


if __name__ == "__main__":
    run_example_tasks()



