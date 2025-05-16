from dotenv import load_dotenv
load_dotenv()

import os
from loguru import logger

from camel.models import ModelFactory
from camel.toolkits import (
    AudioAnalysisToolkit,
    CodeExecutionToolkit,
    DocumentProcessingToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    BrowserToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.toolkits import FunctionTool

from utils import GAIABenchmark


# Configuration
LEVEL = 1
SAVE_RESULT = False
test_idx = [27] # 27


def main():
    """Main function to run the GAIA benchmark."""
    # Create cache directory
    cache_dir = "tmp/"
    os.makedirs(cache_dir, exist_ok=True)

    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "web": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.O3_MINI,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "video": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
        "search": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
        ),
    }
    
    search_toolkit = SearchToolkit()
    
    # Configure toolkits
    tools = [
        *BrowserToolkit(
            headless=True,  # Set to True for headless mode (e.g., on remote servers)
            web_agent_model=models["web"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *DocumentProcessingToolkit().get_tools(),
        *VideoAnalysisToolkit(model=models["video"]).get_tools(),  # This requires OpenAI Key
        *AudioAnalysisToolkit().get_tools(),  # This requires OpenAI Key
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        FunctionTool(search_toolkit.search_google),
        FunctionTool(search_toolkit.search_wiki),
        *ExcelToolkit().get_tools(),
    ]
    
    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Initialize benchmark
    benchmark = GAIABenchmark(
        data_dir="data/gaia",
        save_to=f"results/result.json"
    )

    # Print benchmark information
    print(f"Number of validation examples: {len(benchmark.valid)}")
    print(f"Number of test examples: {len(benchmark.test)}")

    # Run benchmark
    result = benchmark.run(
        on="valid", 
        level=LEVEL, 
        idx=test_idx,
        save_result=SAVE_RESULT,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )

    # Output results
    logger.success(f"Correct: {result['correct']}, Total: {result['total']}")
    logger.success(f"Accuracy: {result['accuracy']}")


if __name__ == "__main__":
    main()
