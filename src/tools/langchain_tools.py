from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
# from langchain.agents import Tool
from langchain_core.tools import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

load_dotenv(override=True)


async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright

def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()

async def other_tools():
    # Python REPL Tool
    python_repl = PythonREPLTool()

    # Wikipedia Tool
    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    # File Tools
    filetools = get_file_tools

    # Search Tool
    serper = GoogleSerperAPIWrapper()
    search_tool = Tool(
        name="websearch",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search."
    )

    return [python_repl, wiki_tool, search_tool] + filetools()

