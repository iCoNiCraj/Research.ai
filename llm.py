import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
import arxiv
from typing import Dict, Union
from exa_py import Exa
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
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
client = arxiv.Client()

system_message = SystemMessage(
    content="""You are a comprehensive research assistant specializing in academic paper retrieval and podcast script creation. Follow this process strictly:
    1. From the user input, identify the research paper URL.
    2. If the URL is from arxiv.org (e.g., contains 'arxiv.org/abs/' or 'arxiv.org/pdf/'), extract the arxiv ID:
       - For URLs like 'https://arxiv.org/abs/<id>', take the ID after '/abs/'.
       - For URLs like 'https://arxiv.org/pdf/<id>.pdf', take the ID between '/pdf/' and '.pdf'.
       Use the extracted ID with the arxiv tool to retrieve the paper's metadata.
    3. If the URL is not from arxiv.org, use the scrape_webpage tool to retrieve the content.
    4. After obtaining the paper's content (either from arxiv or webpage), create a detailed podcast script.

    The podcast script must be in a structured JSON format. The JSON object should have a "title" key containing the research paper's title. All other keys (e.g., "host_intro", "paper_overview", "methodology", "results", "real_world_applications", "limitations", "conclusion", "outro") should contain a list of dialogue objects. The "key_insights" key should contain a list of lists, where each inner list represents one key insight and contains dialogue objects for that insight.

    Each dialogue object within the lists must have the following structure:
    {
        "speaker": "Host 1 (UK)" or "Host 2 (India)",
        "dialogue": "The actual line spoken by the host."
    }

    Example structure for a conversational section:
    "host_intro": [
        { "speaker": "Host 1 (UK)", "dialogue": "Hello and welcome..." },
        { "speaker": "Host 2 (India)", "dialogue": "And I'm Priya..." },
        { "speaker": "Host 1 (UK)", "dialogue": "That sounds remarkable..." }
    ]

    Example structure for key_insights:
    "key_insights": [
        [ // First insight
            { "speaker": "Host 1 (UK)", "dialogue": "Let's dive into the first insight..." },
            { "speaker": "Host 2 (India)", "dialogue": "The first key insight is..." }
        ],
        [ // Second insight
            { "speaker": "Host 1 (UK)", "dialogue": "What's the second major insight?" },
            { "speaker": "Host 2 (India)", "dialogue": "The second key insight involves..." }
        ]
        // ... more insights
    ]


    Guidelines for the podcast script:
    - **Strict Requirement:** The script must represent a natural, flowing conversation between two hosts: Host 1 (with a standard UK accent) and Host 2 (with a standard Indian accent). Ensure natural turn-taking in the dialogue objects list.
    - **Tone:** Maintain a conservative, respectful, and professional tone throughout. Avoid overly casual slang, controversial statements, or inappropriate humor. The focus should be on clear, informative discussion of the research.
    - **Conversational Flow:** Ensure smooth transitions between topics within the dialogue. Use phrases that acknowledge the other speaker and build upon their points.
    - **Engagement:** Use a conversational, engaging style appropriate for an educated audience interested in research. Rhetorical questions can be used by either host to engage the listener.
    - **Content Structure (apply conversationally within the dialogue objects):**
        - For the host_intro list:
          - Start with a hook (e.g., a surprising fact, statistic, or question).
          - Provide background on the research area and introduce the paper (mention authors/date if available).
          - Set the stage and build curiosity.
        - For the paper_overview list:
          - Provide a comprehensive overview: problem, significance, context.
          - Mention notable aspects (novel approaches, impactful findings).
        - For key_insights list of lists:
          - Discuss at least three major insights, each in its own inner list. Explain what, why, and how for each.
          - Use analogies or examples suitable for the conversational format. Include more insights if significant.
        - For methodology list:
          - Explain the methodology conversationally, breaking it down.
          - Discuss the rationale behind methods.
        - For results list:
          - Present key results and interpret their significance.
          - Discuss surprising findings.
        - For real_world_applications list:
          - Provide multiple concrete examples of applications.
          - Explain potential impact.
        - For limitations list:
          - Discuss limitations noted in the paper and their implications.
          - Suggest future research directions.
        - For conclusion list:
          - Summarize main points and key takeaways.
          - Offer brief reflections on the research's impact.
        - For outro list:
          - Thank the listener and encourage further exploration.
          - Tease the next episode.
    - **Speaker Attribution:** Use the "speaker" key in each dialogue object to indicate "Host 1 (UK)" or "Host 2 (India)". Do NOT include speaker tags within the "dialogue" text itself.
    - **Length:** Ensure the total dialogue across all sections is detailed and comprehensive, aiming for at least 1500 words to make the podcast at least 10 minutes long when read aloud at a normal pace.
    - **Clarity:** Use clear language in the "dialogue" text and explain complex concepts accessibly, using analogies or metaphors where helpful.
    - **Accuracy:** Base the script strictly on the information retrieved from the research paper (via arxiv tool or scraping).
    - **Output Format:** Ensure the final output is ONLY the valid JSON object described above, enclosed in ```json ... ``` markers.
    """
)

prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="messages")
])

@tool
def search_arxiv(query: str) -> Union[Dict[str, str], str]:
    """Retrieve a paper from arxiv given an ID."""
    try:
        if not re.match(r'\d+\.\d+', query):
            return "Invalid arxiv ID format. Please provide a valid ID (e.g., 2504.20010)."
        search = arxiv.Search(id_list=[query])
        paper = next(client.results(search), None)
        if paper:
            return {
                "title": paper.title,
                "summary": paper.summary,
                "authors": [author.name for author in paper.authors],
                "url": paper.entry_id
            }
        return "No paper found with that ID."
    except Exception as e:
        return f"Error searching arxiv: {str(e)}"

@tool
def scrape_webpage(url: str) -> Union[Dict[str, str], str]:
    """Scrape content from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        content = "\n".join([element.get_text().strip() for element in main_content if element.get_text().strip()])
        title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled Research Paper"
        
        if not content:
            return "No substantial content found on the webpage."
        
        return {
            "title": title,
            "content": content
        }
    except requests.RequestException as e:
        return f"Error scraping webpage: {str(e)}"
    except Exception as e:
        return f"Unexpected error during scraping: {str(e)}"

tools = [search_arxiv, scrape_webpage]

agent_executor = create_react_agent(model, tools, prompt=prompt, checkpointer=memory)

def process_url(research_paper_url: str) -> str:
    """Process user input to create a podcast script for the research paper."""
    config = {"configurable": {"thread_id": "podcast_script"}}
    messages = [HumanMessage(content=f"Create a podcast script for this research paper: {research_paper_url}")]
    try:
        result = agent_executor.invoke({"messages": messages}, config=config)
        final_message = result.get("messages", [])[-1].content
        match = re.search(r'```json\n(.*?)```', final_message, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            final_message = json_content.replace("```json", "").replace("```", "").strip()
        else:
            final_message = final_message.strip()
        return final_message
    except Exception as e:
        return f"Error generating podcast script: {str(e)}"

if __name__ == "__main__":
    url = input("Enter the research paper URL: ")
    podcast_script = process_url(url)
    print(f"Podcast Script:\n{podcast_script}")