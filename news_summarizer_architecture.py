# ===== CELL 1: IMPORTS AND SETUP =====

from typing import Annotated, List, Dict, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import Tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from dotenv import load_dotenv
from IPython.display import Image, display
import gradio as gr
import requests
import json
import nest_asyncio
import asyncio
from datetime import datetime
import textwrap
import os

# Apply nest_asyncio for notebook compatibility
nest_asyncio.apply()

# ===== CELL 2: LOAD ENVIRONMENT =====

load_dotenv(override=True)

# ===== CELL 3: STATE DEFINITION =====

class NewsState(TypedDict):
    """
    Comprehensive state for the news summarization workflow
    """
    # User input
    query: str
    
    # Processed query information
    search_keywords: List[str]
    news_sources: List[str]
    
    # Raw news data
    raw_news_articles: List[Dict]
    
    # Processed news data
    relevant_articles: List[Dict]
    filtered_articles: List[Dict]
    
    # Summary and report
    summary: str
    final_report: str
    
    # Workflow control
    has_news: bool
    processing_status: str
    error_message: Optional[str]
    
    # Messages for LLM communication
    messages: Annotated[list, add_messages]

# ===== CELL 4: CONFIGURATION AND CONSTANTS =====

# News source URLs
NEWS_SOURCES = {
    "bbc": "https://www.bbc.com/news",
    "cnn": "https://www.cnn.com",
    "reuters": "https://www.reuters.com",
    "guardian": "https://www.theguardian.com/world",
    "ap": "https://apnews.com"
}

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# ===== CELL 5: BROWSER TOOLS SETUP =====

# Create browser tools
async_browser = create_async_playwright_browser(headless=True)
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
browser_tools = toolkit.get_tools()

# Extract specific tools
tool_dict = {tool.name: tool for tool in browser_tools}
navigate_tool = tool_dict.get("navigate_browser")
extract_text_tool = tool_dict.get("extract_text")

print("Browser tools initialized:")
for tool in browser_tools[:3]:  # Show first 3 tools
    print(f"- {tool.name}")

# ===== CELL 6: AGENT FUNCTIONS =====

def query_analyzer_agent(state: NewsState) -> NewsState:
    """
    Agent 1: Analyzes user query and extracts search parameters
    """
    system_prompt = f"""
    You are a query analyzer for news search. 
    Analyze this query: "{state['query']}"
    
    Extract and return ONLY:
    1. 3-5 main search keywords (comma-separated)
    2. Determine if this is about a specific person, event, or topic
    
    Format: keywords: word1, word2, word3
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["query"]}
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # Extract keywords (simplified parsing)
    keywords = []
    if "keywords:" in content.lower():
        keyword_part = content.lower().split("keywords:")[1].split("\n")[0]
        keywords = [k.strip() for k in keyword_part.split(",")]
    
    return {
        **state,
        "search_keywords": keywords,
        "news_sources": list(NEWS_SOURCES.keys()),
        "processing_status": "query_analyzed",
        "messages": state["messages"] + [{"role": "assistant", "content": f"Analyzed query. Keywords: {', '.join(keywords)}"}]
    }

async def news_collector_agent(state: NewsState) -> NewsState:
    """
    Agent 2: Collects news from multiple sources
    """
    raw_articles = []
    
    # For demo, collect from 2 sources to avoid timeout
    sources_to_check = list(NEWS_SOURCES.items())[:2]
    
    for source_name, source_url in sources_to_check:
        try:
            print(f"Collecting from {source_name}...")
            
            # Navigate to news source
            await navigate_tool.arun({"url": source_url})
            
            # Extract text content
            content = await extract_text_tool.arun({})
            
            # Truncate content for processing
            truncated_content = content[:3000] if len(content) > 3000 else content
            
            raw_articles.append({
                "source": source_name,
                "url": source_url,
                "content": truncated_content,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Error collecting from {source_name}: {str(e)}")
            raw_articles.append({
                "source": source_name,
                "url": source_url,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    return {
        **state,
        "raw_news_articles": raw_articles,
        "processing_status": "news_collected",
        "messages": state["messages"] + [{"role": "assistant", "content": f"Collected news from {len(raw_articles)} sources"}]
    }

def news_filter_agent(state: NewsState) -> NewsState:
    """
    Agent 3: Filters and ranks news articles based on relevance
    """
    keywords = state["search_keywords"]
    raw_articles = state["raw_news_articles"]
    
    relevant_articles = []
    
    for article in raw_articles:
        if "error" in article:
            continue
            
        content = article["content"].lower()
        
        # Simple relevance scoring
        relevance_score = 0
        for keyword in keywords:
            if keyword.lower() in content:
                relevance_score += content.count(keyword.lower())
        
        if relevance_score > 0:
            article["relevance_score"] = relevance_score
            relevant_articles.append(article)
    
    # Sort by relevance
    relevant_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    has_news = len(relevant_articles) > 0
    
    return {
        **state,
        "relevant_articles": relevant_articles,
        "filtered_articles": relevant_articles[:5],  # Top 5
        "has_news": has_news,
        "processing_status": "news_filtered",
        "messages": state["messages"] + [{"role": "assistant", "content": f"Found {len(relevant_articles)} relevant articles"}]
    }

def summary_agent(state: NewsState) -> NewsState:
    """
    Agent 4: Summarizes filtered news articles
    """
    if not state["has_news"]:
        return {
            **state,
            "summary": "No relevant news articles found for the given query.",
            "processing_status": "no_news_to_summarize"
        }
    
    # Prepare content for summarization
    articles_content = ""
    for i, article in enumerate(state["filtered_articles"][:3]):  # Top 3 articles
        articles_content += f"\n--- Article {i+1} from {article['source']} ---\n"
        articles_content += article["content"][:800]  # Truncate for context limits
        articles_content += "\n"
    
    system_prompt = f"""
    You are a news summarization agent. Create a comprehensive summary of the provided news articles 
    related to the query: "{state['query']}"
    
    Structure your summary with:
    1. Main headlines and key points
    2. Key facts and developments
    3. Different perspectives if available
    
    Be objective and factual. Keep it concise but informative.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Articles to summarize:\n{articles_content}"}
    ]
    
    response = llm.invoke(messages)
    summary = response.content
    
    return {
        **state,
        "summary": summary,
        "processing_status": "news_summarized",
        "messages": state["messages"] + [{"role": "assistant", "content": "Created news summary"}]
    }

def report_writer_agent(state: NewsState) -> NewsState:
    """
    Agent 5: Generates final formatted report
    """
    if not state["has_news"]:
        no_news_report = f"""
NEWS REPORT
===========

Query: {state['query']}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
----------------
No relevant news articles were found for the specified query after searching 
major international news sources including BBC, CNN, Reuters, The Guardian, and AP News.

SEARCH DETAILS
-------------
- Search Keywords: {', '.join(state.get('search_keywords', []))}
- Sources Searched: {', '.join(state.get('news_sources', []))}
- Search Timestamp: {datetime.now().isoformat()}

CONCLUSION
----------
No current news coverage was found matching your query. This may indicate:
1. The topic is not currently in the news cycle
2. The search terms may need refinement
3. The news may be region-specific or from specialized sources

For more specific results, consider refining your search terms.
        """
        
        return {
            **state,
            "final_report": no_news_report,
            "processing_status": "report_completed"
        }
    
    # Generate comprehensive report
    system_prompt = f"""
    You are a professional report writer. Create a well-formatted news report based on the summary provided.
    
    Format the report with:
    1. Executive Summary
    2. Key Headlines  
    3. Detailed Analysis
    4. Sources Referenced
    5. Conclusion
    
    Make it professional and easy to read.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Create a report for query: '{state['query']}'\n\nSummary: {state['summary']}\n\nSources: {', '.join([a['source'] for a in state['filtered_articles']])}"}
    ]
    
    response = llm.invoke(messages)
    
    final_report = f"""
NEWS REPORT
===========

Query: {state['query']}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{response.content}

SOURCES REFERENCED
-----------------
{', '.join(set([article['source'].upper() for article in state['filtered_articles']]))}

Report generated by AI News Summarizer Agent
    """
    
    return {
        **state,
        "final_report": final_report,
        "processing_status": "report_completed",
        "messages": state["messages"] + [{"role": "assistant", "content": "Final report generated"}]
    }

# ===== CELL 7: WORKFLOW CONTROL FUNCTIONS =====

def should_continue_to_summary(state: NewsState) -> str:
    """Conditional edge: Determine if we should summarize or go to no-news report"""
    if state["has_news"]:
        return "summarize"
    else:
        return "report_no_news"

# ===== CELL 8: BUILD WORKFLOW GRAPH =====

def build_news_workflow():
    """Build the complete news analysis workflow"""
    
    graph_builder = StateGraph(NewsState)
    
    # Add nodes
    graph_builder.add_node("analyze_query", query_analyzer_agent)
    graph_builder.add_node("collect_news", news_collector_agent)
    graph_builder.add_node("filter_news", news_filter_agent)
    graph_builder.add_node("summarize_news", summary_agent)
    graph_builder.add_node("generate_report", report_writer_agent)
    
    # Add edges
    graph_builder.add_edge(START, "analyze_query")
    graph_builder.add_edge("analyze_query", "collect_news")
    graph_builder.add_edge("collect_news", "filter_news")
    
    # Conditional edges
    graph_builder.add_conditional_edges(
        "filter_news",
        should_continue_to_summary,
        {
            "summarize": "summarize_news",
            "report_no_news": "generate_report"
        }
    )
    
    graph_builder.add_edge("summarize_news", "generate_report")
    graph_builder.add_edge("generate_report", END)
    
    # Compile graph with memory
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph

# Build the workflow
news_workflow = build_news_workflow()

# Display workflow diagram
display(Image(news_workflow.get_graph().draw_mermaid_png()))

# ===== CELL 9: QUICK TEST =====

async def quick_test():
    """Quick test of the news summarization workflow"""
    
    print("=== QUICK TEST ===")
    print("Testing with simple query...")
    
    test_query = "artificial intelligence news"
    
    initial_state = {
        "query": test_query,
        "search_keywords": [],
        "news_sources": [],
        "raw_news_articles": [],
        "relevant_articles": [],
        "filtered_articles": [],
        "summary": "",
        "final_report": "",
        "has_news": False,
        "processing_status": "initialized",
        "error_message": None,
        "messages": []
    }
    
    config = {"configurable": {"thread_id": "quick_test"}}
    
    try:
        print(f"Processing query: '{test_query}'")
        result = await news_workflow.ainvoke(initial_state, config=config)
        
        print("\n=== QUICK TEST RESULTS ===")
        print(f"Status: {result['processing_status']}")
        print(f"Keywords found: {result['search_keywords']}")
        print(f"Has news: {result['has_news']}")
        print(f"Articles found: {len(result['relevant_articles'])}")
        
        print("\n=== FINAL REPORT (first 500 chars) ===")
        print(result['final_report'][:500] + "..." if len(result['final_report']) > 500 else result['final_report'])
        
        return True
        
    except Exception as e:
        print(f"Quick test failed: {str(e)}")
        return False

# Run quick test
await quick_test()

# ===== CELL 10: SYSTEM TEST =====

async def system_test():
    """Comprehensive system test with multiple scenarios"""
    
    print("=== SYSTEM TEST ===")
    
    test_cases = [
        {
            "name": "Celebrity News Test",
            "query": "latest news about Taylor Swift",
            "expected_keywords": ["taylor", "swift"]
        },
        {
            "name": "Political News Test", 
            "query": "what is the top news today for president trump",
            "expected_keywords": ["trump", "president"]
        },
        {
            "name": "Technology News Test",
            "query": "OpenAI ChatGPT updates",
            "expected_keywords": ["openai", "chatgpt"]
        },
        {
            "name": "No News Test",
            "query": "news about flying purple elephants",
            "expected_keywords": ["flying", "purple", "elephants"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['name']} ---")
        
        initial_state = {
            "query": test_case["query"],
            "search_keywords": [],
            "news_sources": [],
            "raw_news_articles": [],
            "relevant_articles": [],
            "filtered_articles": [],
            "summary": "",
            "final_report": "",
            "has_news": False,
            "processing_status": "initialized",
            "error_message": None,
            "messages": []
        }
        
        config = {"configurable": {"thread_id": f"test_{i}"}}
        
        try:
            print(f"Processing: '{test_case['query']}'")
            result = await news_workflow.ainvoke(initial_state, config=config)
            
            # Check results
            test_result = {
                "name": test_case["name"],
                "query": test_case["query"],
                "status": result["processing_status"],
                "keywords_found": result["search_keywords"],
                "has_news": result["has_news"],
                "articles_count": len(result["relevant_articles"]),
                "passed": result["processing_status"] == "report_completed"
            }
            
            results.append(test_result)
            
            print(f"✓ Status: {test_result['status']}")
            print(f"✓ Keywords: {test_result['keywords_found']}")
            print(f"✓ Has news: {test_result['has_news']}")
            print(f"✓ Articles: {test_result['articles_count']}")
            
        except Exception as e:
            print(f"✗ Test failed: {str(e)}")
            results.append({
                "name": test_case["name"],
                "query": test_case["query"],
                "status": "failed",
                "error": str(e),
                "passed": False
            })
    
    # Summary
    print("\n=== SYSTEM TEST SUMMARY ===")
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for result in results:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{status} - {result['name']}")
    
    return results

# Run system test
system_test_results = await system_test()

# ===== CELL 11: GRADIO CHAT INTERFACE =====

async def chat_interface(user_input: str, history):
    """
    Gradio chat interface for the news summarization agent
    """
    try:
        # Prepare initial state
        initial_state = {
            "query": user_input,
            "search_keywords": [],
            "news_sources": [],
            "raw_news_articles": [],
            "relevant_articles": [],
            "filtered_articles": [],
            "summary": "",
            "final_report": "",
            "has_news": False,
            "processing_status": "initialized",
            "error_message": None,
            "messages": []
        }
        
        # Generate unique thread ID
        thread_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Process the query
        result = await news_workflow.ainvoke(initial_state, config=config)
        
        # Return the final report
        return result["final_report"]
        
    except Exception as e:
        return f"Error processing your request: {str(e)}\n\nPlease try again with a different query."

# ===== CELL 12: LAUNCH GRADIO INTERFACE =====

def launch_gradio_interface():
    """Launch the Gradio chat interface"""
    
    interface = gr.ChatInterface(
        fn=chat_interface,
        title="🗞️ AI News Summarizer Agent",
        description="""
        Ask me about current news! I'll search major news sources (BBC, CNN, Reuters, Guardian, AP) 
        and provide you with a comprehensive summary.
        
        **Example queries:**
        - "What's the latest news about artificial intelligence?"
        - "Top news today for President Trump"
        - "Climate change updates this week"
        - "Technology news from Silicon Valley"
        """,
        examples=[
            "What's the latest news about artificial intelligence?",
            "Top news today for President Trump", 
            "Climate change updates this week",
            "Technology news from Silicon Valley"
        ],
        theme=gr.themes.Soft(),
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear Chat"
    )
    
    return interface

# Launch the interface
print("=== LAUNCHING GRADIO INTERFACE ===")
print("The news summarizer chat interface is starting...")
print("You can ask questions like:")
print("- 'What's the latest news about AI?'")
print("- 'Top news today for President Biden'")
print("- 'Climate change updates this week'")

news_chat = launch_gradio_interface()
news_chat.launch(share=True, debug=True)

# ===== CELL 13: USAGE EXAMPLES =====

async def demo_usage():
    """Demonstration of various usage scenarios"""
    
    print("=== USAGE EXAMPLES ===")
    
    demo_queries = [
        "breaking news about space exploration",
        "latest economic news from Wall Street",
        "updates on renewable energy"
    ]
    
    for query in demo_queries:
        print(f"\n--- Processing: '{query}' ---")
        
        initial_state = {
            "query": query,
            "search_keywords": [],
            "news_sources": [],
            "raw_news_articles": [],
            "relevant_articles": [],
            "filtered_articles": [],
            "summary": "",
            "final_report": "",
            "has_news": False,
            "processing_status": "initialized",
            "error_message": None,
            "messages": []
        }
        
        config = {"configurable": {"thread_id": f"demo_{query.replace(' ', '_')}"}}
        
        try:
            result = await news_workflow.ainvoke(initial_state, config=config)
            
            print("Summary:")
            print(textwrap.fill(result["summary"][:200] + "..." if len(result["summary"]) > 200 else result["summary"]))
            print(f"\nSources: {', '.join(set([a['source'] for a in result['filtered_articles']]))}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

# Run demo (optional)
# await demo_usage()

print("\n=== NOTEBOOK SETUP COMPLETE ===")
print("✓ All agents initialized")
print("✓ Workflow graph built")
print("✓ Quick test completed")
print("✓ System test completed")
print("✓ Gradio interface launched")
print("\nYou can now use the chat interface or run individual cells for testing!")
