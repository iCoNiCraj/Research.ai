import os
import re
import requests
import json
from dotenv import load_dotenv
load_dotenv()
import arxiv
from typing import List, Dict, Any
from exa_py import Exa
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import difflib

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
            max_results=60,
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
        return papers
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
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=10&fields=title,year,authors,abstract,url,openAccessPdf,paperId,citationCount,venue"
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
    
def is_url(input_string: str) -> bool:
    """Check if the input string is a URL."""
    url_pattern = r'^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$'
    return bool(re.match(url_pattern, input_string))

def process_input(user_input: str) -> str:
    """Process user input to retrieve the top 10 research papers as a JSON array."""
    exa_results = None
    print(f"User input: {user_input}")
    if is_url(user_input):
        message = f"Find research papers related to the paper at {user_input}."
        exa_results = find_similar_and_contents.invoke(user_input)
    else:
        message = f"Find research papers related to: {user_input}."
        exa_results = search_and_contents.invoke(message)
    try:
        title = None
        if exa_results:
            title_prompt = "Write a clean title to search on arxiv or semantic search from this text. Return ONLY the title, no quotes, no prefixes, no bullet points, no special characters only neatly spaced words as the most apt title for: " + str(exa_results)
            title_response = model.invoke(title_prompt)
            raw_title = title_response.content.strip()
            if raw_title:
                clean_title = re.sub(r'[^\w\s]', '', raw_title).strip()
                title = clean_title.split('\n')[0].strip()
                print(f"Generated title: {title}")
        if not title:
            title = user_input.strip()
        arxiv_results = search_arxiv.invoke(title)
        semantic_scholar_results = search_semantic_scholar.invoke(title)
        papers = arxiv_results + semantic_scholar_results
        formatted_papers = []
        for paper in papers:
            formatted_paper = {
                "title": paper.get("title", "Unknown Title"),
                "authors": paper.get("authors", []),
                "year": paper.get("year", paper.get("published", 0)),
                "abstract": paper.get("abstract", paper.get("summary", "No abstract available")),
                "url": paper.get("url", paper.get("pdf_url", "")),
                "source": paper.get("source", "unknown"),
                "categories": paper.get("categories", []),
                "id": paper.get("paperId", paper.get("arxiv_id", ""))
            }
            year = formatted_paper.get("year")
            formatted_papers.append(formatted_paper)
        seen_ids = set()
        seen_titles = set()
        deduplicated_papers = []
        for paper in formatted_papers:
            paper_id = paper.get("id")
            paper_title = paper.get("title")
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                if paper_title and paper_title not in seen_titles:
                    seen_titles.add(paper_title)
                    deduplicated_papers.append(paper)
        close_matches = difflib.get_close_matches(title, [paper["title"] for paper in deduplicated_papers], n=10)
        relevant_papers = []
        for paper in deduplicated_papers:
            if paper["title"] in close_matches:
                relevant_papers.append(paper)
        relevant_papers = sorted(relevant_papers, key=lambda x: x.get("year", 0), reverse=True)
        return relevant_papers
    except Exception as e:
        return json.dumps([{"error": f"Error during processing: {str(e)}"}])