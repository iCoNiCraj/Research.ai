import os
import re
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import arxiv
from typing import List, Dict, Any
from exa_py import Exa
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

memory = MemorySaver()
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

try:
    exa = Exa(api_key=os.environ["EXA_API_KEY"])
except KeyError:
    print("Warning: EXA_API_KEY environment variable not found")
    exa = None

@tool
def search_and_contents(query: str) -> Dict[str, Any]:
    """Search for webpages based on the query and retrieve their contents."""
    if not exa:
        return {"error": "Exa API key not configured"}
    try:
        results = exa.search_and_contents(
            query, use_autoprompt=True, num_results=3, text=True, highlights=True
        )
        return {"status": "success", "results": results, "source": "exa"}
    except Exception as e:
        return {"error": f"Exa search failed: {str(e)}"}

@tool
def find_similar_and_contents(url: str) -> Dict[str, Any]:
    """Search for webpages similar to a given URL and retrieve their contents."""
    if not exa:
        return {"error": "Exa API key not configured"}
    try:
        results = exa.find_similar_and_contents(
            url, num_results=3, text=True, highlights=True
        )
        return {"status": "success", "results": results, "source": "exa"}
    except Exception as e:
        return {"error": f"Exa similar search failed: {str(e)}"}

@tool
def search_arxiv(query: str) -> List[Dict[str, Any]]:
    """Search for academic papers from arXiv related to the query."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "authors": [a.name for a in result.authors],
                "published": result.published.isoformat(),
                "year": result.published.year,
                "arxiv_id": result.entry_id.split('/')[-1],
                "categories": result.categories,
                "source": "arxiv"
            })
        return papers[:10]
    except Exception as e:
        return [{"error": f"arXiv search failed: {str(e)}"}]

@tool
def search_semantic_scholar(query: str) -> List[Dict[str, Any]]:
    """Search for academic papers from Semantic Scholar related to the query."""
    try:
        papers = []
        headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "cache-control": "max-age=0",
            "priority": "u=0, i",
            "sec-ch-ua": '"Microsoft Edge";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": '"Android"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 Edg/135.0.0.0"
        }
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=10&sort=year&fields=title,year,authors,abstract,url,openAccessPdf,paperId,citationCount,venue"
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        raw_results = data.get("data", [])
        for result in raw_results:
            paper_data = {
                "title": result.get("title", "No title available"),
                "summary": result.get("abstract", "No abstract available"),
                "authors": result.get("authors", []),
                "year": result.get("year"),
                "url": result.get("url"),
                "pdf_url": result.get("openAccessPdf", {}).get("url") if result.get("openAccessPdf") else None,
                "paperId": result.get("paperId"),
                "citationCount": result.get("citationCount"),
                "venue": result.get("venue"),
                "source": "semantic_scholar",
                "categories": result.get("fieldsOfStudy", [])
            }
            papers.append(paper_data)
        return papers
    except Exception as e:
        return [{"error": f"Semantic Scholar search failed: {str(e)}"}]

tools = [search_and_contents, find_similar_and_contents, search_arxiv, search_semantic_scholar]

system_message = SystemMessage(
    content="""You are a comprehensive research assistant specializing in academic paper retrieval. Follow this process strictly:

1. First, use search_and_contents tool to get initial information if the input is a query or find_similar_and_contents if it's a URL.
2. After analyzing the Exa results, create a query which will be the most apt title for the human message to search for related papers using BOTH search_arxiv AND search_semantic_scholar - use the same query for both to ensure comprehensive information gathering .
3. IMPORTANT: Do not spend excessive time analyzing each paper in detail. Focus on efficiently gathering results from all sources first.
4. After collecting the papers from arXiv and Semantic Scholar, your task is complete. Do not attempt to compile or format the results further.
5. If any tool encounters an error, acknowledge it but continue with the available information from other tools.
6. Work sequentially and efficiently: Exa → arXiv AND Semantic Scholar.

The final compilation of the top 10 papers will be handled separately."""
)

prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="messages")
])

agent_executor = create_react_agent(model, tools, prompt=prompt, checkpointer=memory)

def is_url(input_string: str) -> bool:
    """Check if the input string is a URL."""
    url_pattern = r'^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$'
    return bool(re.match(url_pattern, input_string))

def process_input(user_input: str, config: dict = None) -> str:
    """Process user input to retrieve the top 10 research papers as a JSON array."""
    if config is None:
        config = {"configurable": {"thread_id": "research_assistant"}}
    
    if is_url(user_input):
        messages = [HumanMessage(content=f"Find research papers related to the paper at {user_input}. Begin with find_similar_and_contents, then search both arXiv and Semantic Scholar.")]
    else:
        messages = [HumanMessage(content=f"Find research papers related to: {user_input}. Begin with search_and_contents, then search both arXiv and Semantic Scholar.")]
    
    try:
        papers = []
        for step in agent_executor.stream(
            {"messages": messages},
            config,
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
            last_message = step["messages"][-1]
            if isinstance(last_message, ToolMessage) and last_message.name in ["search_arxiv"]:
                try:
                    tool_output = json.loads(last_message.content) if isinstance(last_message.content, str) else last_message.content
                    if isinstance(tool_output, list) and all(isinstance(item, dict) for item in tool_output):
                        papers.extend(tool_output)
                except (json.JSONDecodeError, TypeError):
                    continue
        for paper in papers:
            if "published" in paper and paper["published"]:
                try:
                    paper["_sort_date"] = datetime.fromisoformat(paper["published"])
                except ValueError:
                    paper["_sort_date"] = None
            elif "year" in paper and paper["year"]:
                try:
                    year = int(paper["year"])
                    paper["_sort_date"] = datetime(year, 1, 1)
                except ValueError:
                    paper["_sort_date"] = None
            else:
                paper["_sort_date"] = None
        
        # Sort by date (newest first), with papers lacking dates at the end
        sorted_papers = sorted(
            papers,
            key=lambda x: (x["_sort_date"] is not None, x["_sort_date"]),
            reverse=True
        )
        
        # Select top 10
        top_papers = sorted_papers[:10]
        
        # Format into standardized JSON
        formatted_papers = []
        for paper in top_papers:
            formatted_paper = {
                "title": paper.get("title", "Unknown Title"),
                "authors": paper.get("authors", []),
                "year": paper.get("year", paper.get("published", "Unknown Year")),
                "abstract": paper.get("abstract", paper.get("summary", "No abstract available")),
                "url": paper.get("url", paper.get("pdf_url", "")),
                "source": paper.get("source", "unknown"),
                "categories": paper.get("categories", []),
                "id": paper.get("paperId", paper.get("arxiv_id", ""))
            }
            formatted_papers.append(formatted_paper)
        
        return formatted_papers
    except Exception as e:
        return json.dumps([{"error": f"Error during processing: {str(e)}"}])