import os, datetime, json
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from typing import Annotated, Optional
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image
from langchain_community.tools import TavilySearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage
import yfinance as yf


# ANSI escape codes for color
RED = "\033[31m"
RESET = "\033[0m"

def print_message(title, content):
    print(RED + "**** " + title + " ****" + RESET + "\n" + content + "\n")

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = AzureChatOpenAI(model="gpt-4o-mini", temperature=0, api_version=os.environ['AZURE_OPENAI_API_VERSION'])

class ResearchState(MessagesState):
    ResearchGoal: str


# fake weather search
def weather_search(query: Annotated[str, "The search query to run"]) -> Annotated[str, "The search results"]:
    """Search the weather with the query"""
    print_message("Weather Query", query)
    result = "The weather is sunny and 75 degrees."
    print_message("Weather Results", result)
    return result


# stock search tool using Yahoo Finance
def stock_search(symbol: Annotated[str, "The stock ticker symbol (e.g., 'AAPL' for Apple, 'GOOGL' for Alphabet)"], start_date: Annotated[str, "The start date of the data being requested in YYYY-MM-DD format"], end_date: Annotated[str, "The end date of the data being requested in YYYY-MM-DD format"]) -> Annotated[Optional[list[dict]], "A list of dictionaries where each dictionary represents a row of stock data with keys like 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', etc."]:
    """
    Retrieve stock price data for the given stock symbol over a specified date range using Yahoo Finance.
    """
    print_message("Stock Query", f"Symbol: {symbol}, Start Date: {start_date}, End Date: {end_date}")
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)  # Get data for the date range
        
        if not data.empty:
            # Convert the DataFrame to a list of dictionaries
            data.reset_index(inplace=True)  # Make 'Date' a column instead of the index
            data_list = data.to_dict(orient='records')  # Convert to list of dicts
            try:
                # Fix the unseralizable datetime objects
                for row in data_list:
                    row["Date"] = row["Date"].strftime("%Y-%m-%d")
                print_message("Stock Query results", json.dumps(data_list))
            except Exception as e:
                print_message("Error", "Error converting query results to JSON: " + str(e))
            return data_list
        else:
            raise ValueError("No data found for the provided symbol and date range.")
    except Exception as e:
        print(f"Error retrieving stock price data: {e}")
        return None


# web search tool using Tavily
def tavily_search(query: Annotated[str, "The search query to run"]) -> Annotated[TavilySearchResults, "The search results"]:
    """Search the web with the query"""
    print_message("Web Query", query)
    # create a Tavily search object
    search = TavilySearchResults()
    # run the search
    result = search.invoke({"query": query})
    try:
        print_message("Web Search results", json.dumps(result))
    except:
        print_message("Error", "Error converting search results to JSON")

    return result


# Wikipedia search tool
def wikipedia_search(query: Annotated[str, "The search query to run"]) -> Annotated[str, "The search results"]:
    """Search Wikipedia with the query"""
    print_message("Wikipedia Query", query)
    # create a Wikipedia search object
    search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    # run the search
    result = search.run(query)
    try:
        print_message("Wikipedia Search Results", json.dumps(result))
    except:
        print_message("Error", "Error converting search results to JSON")

    return result


tools = [tavily_search, wikipedia_search, weather_search, stock_search]
llm_with_tools = model.bind_tools(tools)


def research_agent(state: ResearchState):
    # extract the research goal from the state
    research_goal = state['ResearchGoal']

    # set the system message
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    system_message = """Today is {today}.  Use the available tools to research the following: {research_goal}.
    Only use the tools necessary to find the informtion needed and provide the results.  
    Do not use any more tools than are needed.
    You should not share your opinion on the information you get back from the tools, you should only 
    synthesize and summarize the information as it is given to you.""".format(today=today, research_goal=research_goal)

    # call the LLM with the tools
    llm_response = llm_with_tools.invoke([system_message] + state["messages"])

    # display for demo purposes
    try:
        if llm_response.content == "" and llm_response.tool_calls:
            print_message("LLM Tool Calls", json.dumps(llm_response.tool_calls))
        else:
            print_message("LLM Response", llm_response.content)
    except:
        print_message("Error", "Error converting LLM response to JSON")

    # update the state with the LLM response
    return {"messages": [llm_response]}


graph = StateGraph(ResearchState)
graph.add_node("tools", ToolNode(tools))
graph.add_node("research_agent", research_agent)

graph.add_edge(START, "research_agent")
graph.add_conditional_edges("research_agent", tools_condition, ["tools", END])
graph.add_edge("tools", "research_agent")

app = graph.compile()

# get the current running folder
running_folder = os.path.dirname(os.path.abspath(__file__))

# set the current folder to the research folder
research_folder = running_folder + "/research_agent"

# create the research folder if it does not exist
if not os.path.exists(research_folder):
    os.makedirs(research_folder)

# save the png of the graph
image = Image(app.get_graph(xray=1).draw_mermaid_png())
open(f"{research_folder}/research_agent_graph.png", "wb").write(image.data)

# research_goal = "What was the topic of Present Russell M. Nelson's most recent message during General Conference?"
# research_goal = "Who is President Russell M. Nelson?"
# research_goal = "What is the weather in Salt Lake City?"
research_goal = "How has Apple stock done this week?  What news is impacting the stock price?"

final_state = app.invoke(
    {"messages": [HumanMessage(content=research_goal)], "ResearchGoal": research_goal},
    config={"configurable": {"thread_id": 42}, "recursion_limit": 10}
)

print_message("Final Response", final_state["messages"][-1].content)

