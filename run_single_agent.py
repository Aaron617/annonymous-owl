from dotenv import load_dotenv
load_dotenv(override=True)

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

from utils import GAIABenchmark, OwlChatAgent

# Configuration
LEVEL = 1
SAVE_RESULT = True
test_idx = [0, 1, 2]
MAX_TRIES = 3


def main():
    """Main function to run the GAIA benchmark."""
    # Create cache directory
    cache_dir = "tmp/"
    os.makedirs(cache_dir, exist_ok=True)
    
    image_analysis_model = ModelFactory.create( 
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0},
    )
    
    web_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0},
    )
    
    planning_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.O3_MINI,
        model_config_dict={"temperature": 0},
    )

    search_toolkit = SearchToolkit()
    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp")
    image_analysis_toolkit = ImageAnalysisToolkit(model=image_analysis_model)
    video_analysis_toolkit = VideoAnalysisToolkit(download_directory="tmp/video")
    # audio_analysis_toolkit = AudioAnalysisToolkit(cache_dir="tmp/audio", audio_reasoning_model=audio_reasoning_model)
    audio_analysis_toolkit = AudioAnalysisToolkit(cache_dir="tmp/audio", reasoning=True)
    # video_analysis_toolkit = VideoAnalysisToolkit(download_directory="tmp/video", model=video_analysis_model, use_audio_transcription=True)
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
    browser_simulator_toolkit = BrowserToolkit(headless=True, cache_dir="tmp/browser", planning_agent_model=planning_agent_model, web_agent_model=web_agent_model)
    excel_toolkit = ExcelToolkit()
    
    # Configure toolkits
    tools = [
        search_toolkit.search_google,
        search_toolkit.search_wiki,
        search_toolkit.search_wiki_revisions,
        document_processing_toolkit.extract_document_content,
        browser_simulator_toolkit.browse_url,
        video_analysis_toolkit.ask_question_about_video,
        audio_analysis_toolkit.ask_question_about_audio,
        code_runner_toolkit.execute_code,
        image_analysis_toolkit.ask_question_about_image,
        excel_toolkit.extract_excel_content,
    ]
    
    gaia_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0},
    )
    
    # Initialize benchmark
    benchmark = GAIABenchmark(
        data_dir="data/gaia",
        save_to=f"results/single_agent/level_{LEVEL}_pass_{MAX_TRIES}.json"
    )

    # Print benchmark information
    print(f"Number of validation examples: {len(benchmark.valid)}")
    print(f"Number of test examples: {len(benchmark.test)}")
    
    
    agent = OwlChatAgent(
        """
You are a helpful assistant that can solve complex tasks. Keep in mind that:
Keep in mind that:
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.  
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.  
- You can also simulate browser actions to get more information or verify the information you have found.
""",
        model=gaia_model,
        tools=tools,
    )
    

    # Run benchmark
    result = benchmark.run_single_agent_with_retry(
        agent=agent,
        on="valid", 
        level=LEVEL, 
        idx=test_idx,
        save_result=SAVE_RESULT,
        max_tries=MAX_TRIES,
    )

    # Output results
    logger.success(f"Correct: {result['correct']}, Total: {result['total']}")
    logger.success(f"Accuracy: {result['accuracy']}")


if __name__ == "__main__":
    main()
