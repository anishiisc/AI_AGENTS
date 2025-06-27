#!/usr/bin/env python3
"""
COMPLETE INTEGRATED COMMODITY ANALYSIS SYSTEM
Combines existing geopolitical analysis with natural language interface

INTEGRATION APPROACH: Single file with all components
This demonstrates how to add natural language capabilities to your existing system
"""

# ==================== ORIGINAL IMPORTS (Unchanged) ====================
from agents import Agent, WebSearchTool, trace, Runner, gen_trace_id, function_tool
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
import os
from typing import Dict, List, Optional, Union
from IPython.display import display, Markdown
from datetime import datetime
import json
import re
from enum import Enum

load_dotenv(override=True)

# ==================== ORIGINAL DATA MODELS (Unchanged) ====================
class GeopoliticalSearchItem(BaseModel):
    reason: str = Field(description="Strategic reasoning for why this geopolitical search is critical for commodity price analysis.")
    query: str = Field(description="The specific search term focusing on geopolitical events, wars, policies, or world events.")
    priority: str = Field(description="Priority level: 'high', 'medium', or 'low' based on expected impact on commodity prices.")

class GeopoliticalSearchPlan(BaseModel):
    commodity: str = Field(description="The commodity being analyzed (oil, gold, copper, etc.)")
    searches: List[GeopoliticalSearchItem] = Field(description="List of geopolitical searches to understand global events affecting commodity prices.")

class CommoditySearchItem(BaseModel):
    reason: str = Field(description="Reasoning for this commodity-specific search query.")
    query: str = Field(description="Search term focused on commodity price trends, market analysis, or supply/demand factors.")

class CommoditySearchPlan(BaseModel):
    searches: List[CommoditySearchItem] = Field(description="List of commodity-specific searches for deeper market analysis.")

class CommodityAnalysisReport(BaseModel):
    executive_summary: str = Field(description="2-3 sentence executive summary of key findings and price trend predictions.")
    current_price_trend: str = Field(description="Current price trend analysis (bullish/bearish/neutral) with reasoning.")
    key_geopolitical_factors: List[str] = Field(description="Top 3-5 geopolitical factors currently impacting the commodity.")
    price_drivers: List[str] = Field(description="Primary economic and geopolitical drivers affecting price movements.")
    risk_assessment: str = Field(description="Assessment of geopolitical risks and their potential price impact.")
    market_outlook: str = Field(description="Short to medium-term market outlook and price predictions.")
    html_report: str = Field(description="Complete detailed report in HTML format with proper styling.")
    follow_up_analysis: List[str] = Field(description="Suggested areas for further geopolitical analysis.")

# ==================== NEW DATA MODELS (Added for Natural Language) ====================
class QueryType(Enum):
    PRICE_PREDICTION = "price_prediction"
    GEOPOLITICAL_IMPACT = "geopolitical_impact"
    MARKET_ANALYSIS = "market_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    GENERAL_COMMODITY = "general_commodity"

class QueryIntent(BaseModel):
    commodity: str = Field(description="Primary commodity identified in the query")
    query_type: QueryType = Field(description="Type of analysis requested")
    specific_focus: List[str] = Field(description="Specific aspects user wants analyzed")
    time_horizon: str = Field(description="Time frame for analysis (short-term, medium-term, long-term)")
    additional_commodities: List[str] = Field(default=[], description="Other commodities mentioned for comparison")
    confidence_score: float = Field(description="Confidence in query interpretation (0-1)")

class AnalysisScope(BaseModel):
    primary_commodity: str = Field(description="Main commodity to analyze")
    analysis_type: QueryType = Field(description="Type of analysis to perform")
    focus_areas: List[str] = Field(description="Specific areas to emphasize in analysis")
    comparison_commodities: List[str] = Field(default=[], description="Additional commodities for comparison")
    urgency_level: str = Field(description="Analysis urgency: high, medium, low")

# ==================== ORIGINAL AGENTS (Unchanged) ====================
GEOPOLITICAL_PLANNER_INSTRUCTIONS = """You are a senior geopolitical analyst specializing in commodity markets. 
Given a commodity (oil, gold, copper, etc.), create a strategic search plan to identify current geopolitical events, 
conflicts, policy changes, and world events that could impact the commodity's price.

Focus on:
- Active military conflicts and tensions
- Trade wars and sanctions
- Central bank policies and government decisions  
- Supply chain disruptions
- Political instability in key producing regions
- Environmental and regulatory changes
- International agreements and treaties

Generate 4-5 high-priority searches that will provide comprehensive geopolitical context."""

geopolitical_planner_agent = Agent(
    name="GeopoliticalPlannerAgent",
    instructions=GEOPOLITICAL_PLANNER_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=GeopoliticalSearchPlan,
)

GEOPOLITICAL_RESEARCH_INSTRUCTIONS = """You are a geopolitical research specialist focused on commodity markets.
Analyze web search results for geopolitical events and their potential impact on commodity prices.

Your analysis should be:
- Factual and data-driven
- Focused on price implications
- Concise but comprehensive (2-3 paragraphs max)
- Free of speculation, stick to reported facts
- Include specific dates, locations, and key players when available

Extract the essence of how geopolitical events translate to commodity market pressures."""

geopolitical_research_agent = Agent(
    name="GeopoliticalResearchAgent", 
    instructions=GEOPOLITICAL_RESEARCH_INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)

COMMODITY_PLANNER_INSTRUCTIONS = """You are a commodity market analyst. Based on geopolitical context provided,
create targeted searches to understand current commodity price trends, market sentiment, supply/demand dynamics,
and technical analysis.

Focus on:
- Current price movements and technical indicators
- Supply and demand fundamentals  
- Market sentiment and trader positioning
- Production data and inventory levels
- Expert analyst predictions and forecasts
- Historical price patterns during similar geopolitical events

Generate 3-4 searches for deep commodity market analysis."""

commodity_planner_agent = Agent(
    name="CommodityPlannerAgent",
    instructions=COMMODITY_PLANNER_INSTRUCTIONS, 
    model="gpt-4o-mini",
    output_type=CommoditySearchPlan,
)

COMMODITY_RESEARCH_INSTRUCTIONS = """You are a commodity market research specialist. 
Analyze web search results for commodity price data, market trends, and trading insights.

Your analysis should cover:
- Current price levels and recent movements
- Technical analysis and chart patterns
- Supply/demand fundamentals
- Market sentiment indicators
- Expert forecasts and price targets
- Trading volumes and institutional positioning

Keep analysis factual, quantitative when possible, and focused on actionable market intelligence.
Limit to 2-3 focused paragraphs per search."""

commodity_research_agent = Agent(
    name="CommodityResearchAgent",
    instructions=COMMODITY_RESEARCH_INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],
    model="gpt-4o-mini", 
    model_settings=ModelSettings(tool_choice="required"),
)

REPORT_WRITER_INSTRUCTIONS = """You are a senior commodity analyst preparing reports for institutional clients.
Synthesize geopolitical research and commodity market analysis into a comprehensive, professional report.

Your report must include:
1. Executive Summary with clear price trend prediction
2. Current Market Analysis  
3. Geopolitical Risk Assessment
4. Key Price Drivers Analysis
5. Market Outlook and Recommendations
6. Risk Factors and Scenarios

Format the HTML report with:
- Professional styling with CSS
- Clear headings and sections
- Data tables where appropriate  
- Color coding for risk levels (green=low, yellow=medium, red=high)
- Executive summary highlighted at the top
- Footer with analysis date and disclaimer

Make it visually appealing and easy to scan for busy executives."""

report_writer_agent = Agent(
    name="ReportWriterAgent",
    instructions=REPORT_WRITER_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=CommodityAnalysisReport,
)

@function_tool
def save_html_report(filename: str, html_content: str) -> Dict[str, str]:
    """Save the HTML report to a file"""
    try:
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"reports/{filename}_{timestamp}.html"
        
        with open(full_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return {
            "status": "success", 
            "filename": full_filename,
            "message": f"Report saved successfully to {full_filename}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save report: {str(e)}"
        }

OUTPUT_AGENT_INSTRUCTIONS = """You are responsible for saving analysis reports to HTML files.
You will receive a detailed commodity analysis report and should save it as an HTML file with an appropriate filename.

Use descriptive filenames that include the commodity name and analysis type.
Always confirm successful file creation."""

output_agent = Agent(
    name="OutputAgent",
    instructions=OUTPUT_AGENT_INSTRUCTIONS,
    tools=[save_html_report],
    model="gpt-4o-mini",
)

# ==================== NEW AGENTS (Added for Natural Language) ====================
QUERY_INTERPRETER_INSTRUCTIONS = """You are a natural language query interpreter specializing in commodity and geopolitical analysis requests.

Your task is to analyze user queries and extract:
1. Primary commodity mentioned (oil, gold, copper, wheat, etc.)
2. Type of analysis requested (price prediction, geopolitical impact, market analysis, etc.)
3. Specific focus areas (supply disruption, sanctions impact, weather effects, etc.)
4. Time horizon (short-term: days/weeks, medium-term: months, long-term: quarters/years)
5. Additional commodities for comparison
6. Confidence score in your interpretation

Common query patterns:
- "What will oil prices do given the current situation?" → price_prediction, oil, current geopolitical events
- "How are sanctions affecting gold markets?" → geopolitical_impact, gold, sanctions focus
- "Compare copper vs aluminum in the current market" → comparative_analysis, copper+aluminum
- "What are the risks to wheat supply?" → risk_assessment, wheat, supply chain focus

Be precise in commodity identification and analysis type classification."""

query_interpreter_agent = Agent(
    name="QueryInterpreterAgent",
    instructions=QUERY_INTERPRETER_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=QueryIntent,
)

ANALYSIS_COORDINATOR_INSTRUCTIONS = """You are an analysis coordinator that determines the optimal research strategy based on user query intent.

Based on the interpreted query, decide:
1. Which analysis components are needed (geopolitical, market, comparative)
2. Priority and depth of each analysis type
3. Specific focus areas to emphasize
4. Whether multiple commodities need analysis
5. Urgency level for the analysis

Create a structured analysis plan that will guide the research agents to provide exactly what the user is asking for."""

analysis_coordinator_agent = Agent(
    name="AnalysisCoordinatorAgent",
    instructions=ANALYSIS_COORDINATOR_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=AnalysisScope,
)

CONVERSATIONAL_RESPONSE_INSTRUCTIONS = """You are a conversational commodity analyst that presents technical analysis in an accessible, engaging way.

Transform structured analysis reports into natural language responses that:
1. Directly answer the user's original question
2. Use conversational tone while maintaining professional accuracy
3. Highlight key insights with clear explanations
4. Provide context for technical terms
5. Structure information in logical flow
6. Include relevant caveats and limitations

Response format:
- Start with direct answer to user's question
- Provide supporting analysis in digestible sections
- Use bullet points for key factors
- End with outlook and next steps
- Keep technical jargon minimal but precise"""

conversational_response_agent = Agent(
    name="ConversationalResponseAgent",
    instructions=CONVERSATIONAL_RESPONSE_INSTRUCTIONS,
    model="gpt-4o-mini",
)

# ==================== ORIGINAL ORCHESTRATION FUNCTIONS (Unchanged) ====================
async def plan_geopolitical_searches(commodity: str):
    """Use geopolitical planner to identify key geopolitical events to research"""
    print(f"🌍 Planning geopolitical searches for {commodity}...")
    result = await Runner.run(geopolitical_planner_agent, f"Commodity: {commodity}")
    print(f"📋 Planned {len(result.final_output.searches)} geopolitical searches")
    return result.final_output

async def execute_geopolitical_research(search_plan: GeopoliticalSearchPlan):
    """Execute all geopolitical searches concurrently"""
    print("🔍 Executing geopolitical research...")
    tasks = [asyncio.create_task(geopolitical_search(item)) for item in search_plan.searches]
    results = await asyncio.gather(*tasks)
    print("✅ Completed geopolitical research")
    return results

async def geopolitical_search(item: GeopoliticalSearchItem):
    """Execute individual geopolitical search"""
    input_msg = f"Search Query: {item.query}\nAnalysis Focus: {item.reason}\nPriority: {item.priority}"
    result = await Runner.run(geopolitical_research_agent, input_msg)
    return {
        "query": item.query,
        "priority": item.priority,
        "analysis": result.final_output
    }

async def plan_commodity_searches(commodity: str, geopolitical_context: List[Dict]):
    """Plan commodity-specific searches based on geopolitical context"""
    print(f"📈 Planning commodity market searches for {commodity}...")
    context_summary = "\n".join([f"- {item['query']}: {item['analysis'][:200]}..." for item in geopolitical_context])
    input_msg = f"Commodity: {commodity}\nGeopolitical Context:\n{context_summary}"
    result = await Runner.run(commodity_planner_agent, input_msg)
    print(f"📊 Planned {len(result.final_output.searches)} commodity searches")
    return result.final_output

async def execute_commodity_research(search_plan: CommoditySearchPlan):
    """Execute commodity market research"""
    print("📊 Executing commodity market research...")
    tasks = [asyncio.create_task(commodity_search(item)) for item in search_plan.searches]
    results = await asyncio.gather(*tasks)
    print("✅ Completed commodity research")
    return results

async def commodity_search(item: CommoditySearchItem):
    """Execute individual commodity search"""
    input_msg = f"Search Query: {item.query}\nAnalysis Focus: {item.reason}"
    result = await Runner.run(commodity_research_agent, input_msg)
    return {
        "query": item.query, 
        "analysis": result.final_output
    }

async def generate_analysis_report(commodity: str, geopolitical_results: List[Dict], commodity_results: List[Dict]):
    """Generate comprehensive analysis report"""
    print("📝 Generating comprehensive analysis report...")
    
    input_data = {
        "commodity": commodity,
        "geopolitical_analysis": geopolitical_results,
        "commodity_analysis": commodity_results,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    input_msg = f"""
    Commodity: {commodity}
    Analysis Date: {input_data['analysis_date']}
    
    GEOPOLITICAL RESEARCH FINDINGS:
    {json.dumps(geopolitical_results, indent=2)}
    
    COMMODITY MARKET RESEARCH FINDINGS:  
    {json.dumps(commodity_results, indent=2)}
    
    Please synthesize this into a comprehensive commodity analysis report.
    """
    
    result = await Runner.run(report_writer_agent, input_msg)
    print("✅ Analysis report generated")
    return result.final_output

async def save_report(report: CommodityAnalysisReport, commodity: str):
    """Save the analysis report using output agent"""
    print("💾 Saving analysis report...")
    filename = f"{commodity.lower()}_geopolitical_analysis"
    result = await Runner.run(output_agent, f"Filename: {filename}\nHTML Content: {report.html_report}")
    print("✅ Report saved successfully")
    return result

async def analyze_commodity_geopolitics(commodity: str):
    """
    ORIGINAL MAIN FUNCTION - UNCHANGED
    Main function to execute the complete geopolitical commodity analysis workflow
    """
    
    with trace("Geopolitical Commodity Analysis"):
        print(f"🚀 Starting geopolitical analysis for {commodity.upper()}")
        print("=" * 60)
        
        # Phase 1: Geopolitical Analysis
        print("📍 PHASE 1: GEOPOLITICAL INTELLIGENCE GATHERING")
        geopolitical_plan = await plan_geopolitical_searches(commodity)
        geopolitical_results = await execute_geopolitical_research(geopolitical_plan)
        
        # Phase 2: Commodity Market Analysis  
        print("\n📍 PHASE 2: COMMODITY MARKET ANALYSIS")
        commodity_plan = await plan_commodity_searches(commodity, geopolitical_results)
        commodity_results = await execute_commodity_research(commodity_plan)
        
        # Phase 3: Report Generation
        print("\n📍 PHASE 3: REPORT SYNTHESIS & OUTPUT")
        analysis_report = await generate_analysis_report(commodity, geopolitical_results, commodity_results)
        
        # Phase 4: Save Report
        await save_report(analysis_report, commodity)
        
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETE!")
        print(f"📋 Executive Summary: {analysis_report.executive_summary}")
        print(f"📈 Price Trend: {analysis_report.current_price_trend}")
        
        return analysis_report

# ==================== NEW UTILITY FUNCTIONS (Added for Natural Language) ====================
COMMODITY_KEYWORDS = {
    'oil': ['oil', 'crude', 'petroleum', 'wti', 'brent', 'energy'],
    'gold': ['gold', 'precious metals', 'bullion', 'au'],
    'silver': ['silver', 'ag', 'precious metals'],
    'copper': ['copper', 'cu', 'industrial metals'],
    'wheat': ['wheat', 'grain', 'cereals'],
    'corn': ['corn', 'maize', 'grain'],
    'natural_gas': ['natural gas', 'lng', 'gas', 'natgas'],
    'aluminum': ['aluminum', 'aluminium', 'al', 'bauxite'],
    'iron_ore': ['iron ore', 'iron', 'steel', 'fe'],
    'coal': ['coal', 'thermal coal', 'coking coal'],
    'coffee': ['coffee', 'arabica', 'robusta'],
    'sugar': ['sugar', 'sucrose'],
    'cotton': ['cotton', 'textile'],
    'soybeans': ['soybeans', 'soy', 'oilseeds'],
}

def detect_commodities_in_text(text: str) -> List[str]:
    """Detect commodities mentioned in user text"""
    text_lower = text.lower()
    detected = []
    
    for commodity, keywords in COMMODITY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected.append(commodity)
    
    return detected

def extract_time_indicators(text: str) -> str:
    """Extract time horizon indicators from text"""
    text_lower = text.lower()
    
    short_term_indicators = ['today', 'tomorrow', 'this week', 'next week', 'short term', 'immediate', 'now']
    medium_term_indicators = ['this month', 'next month', 'quarter', 'medium term', 'coming months']
    long_term_indicators = ['this year', 'next year', 'long term', 'future', 'outlook']
    
    if any(indicator in text_lower for indicator in short_term_indicators):
        return "short-term"
    elif any(indicator in text_lower for indicator in medium_term_indicators):
        return "medium-term"
    elif any(indicator in text_lower for indicator in long_term_indicators):
        return "long-term"
    else:
        return "medium-term"  # default

# ==================== NEW ORCHESTRATION FUNCTIONS (Added for Natural Language) ====================

async def process_natural_language_query(user_query: str) -> dict:
    """
    NEW FUNCTION: Process natural language query and route to appropriate analysis
    This function coordinates the natural language processing workflow
    """
    
    with trace("Natural Language Query Processing"):
        print(f"🎯 Processing query: '{user_query}'")
        print("=" * 60)
        
        # Phase 1: Interpret the query
        print("📍 PHASE 1: QUERY INTERPRETATION")
        query_intent = await interpret_user_query(user_query)
        print(f"🎯 Identified commodity: {query_intent.commodity}")
        print(f"🔍 Analysis type: {query_intent.query_type.value}")
        print(f"⚡ Confidence: {query_intent.confidence_score:.2f}")
        
        # Phase 2: Plan analysis approach
        print("\n📍 PHASE 2: ANALYSIS PLANNING")
        analysis_scope = await plan_analysis_approach(query_intent, user_query)
        print(f"📊 Primary focus: {analysis_scope.primary_commodity}")
        print(f"🎯 Analysis type: {analysis_scope.analysis_type.value}")
        
        # Phase 3: Execute targeted analysis
        print("\n📍 PHASE 3: TARGETED ANALYSIS EXECUTION")
        analysis_results = await execute_targeted_analysis(analysis_scope)
        
        # Phase 4: Generate conversational response
        print("\n📍 PHASE 4: RESPONSE GENERATION")
        conversational_response = await generate_conversational_response(
            user_query, query_intent, analysis_results
        )
        
        print("\n" + "=" * 60)
        print("🎉 QUERY PROCESSING COMPLETE!")
        
        return {
            "original_query": user_query,
            "query_intent": query_intent,
            "analysis_scope": analysis_scope,
            "technical_analysis": analysis_results,
            "conversational_response": conversational_response,
            "confidence_score": query_intent.confidence_score
        }

async def interpret_user_query(user_query: str) -> QueryIntent:
    """NEW FUNCTION: Interpret user's natural language query"""
    print("🧠 Interpreting user intent...")
    
    # Add basic commodity detection as context
    detected_commodities = detect_commodities_in_text(user_query)
    time_horizon = extract_time_indicators(user_query)
    
    context = f"""
    User Query: {user_query}
    
    Detected Commodities: {detected_commodities}
    Implied Time Horizon: {time_horizon}
    
    Please interpret this query and extract the analysis intent.
    """
    
    result = await Runner.run(query_interpreter_agent, context)
    print(f"✅ Query interpreted with {result.final_output.confidence_score:.2f} confidence")
    return result.final_output

async def plan_analysis_approach(query_intent: QueryIntent, original_query: str) -> AnalysisScope:
    """NEW FUNCTION: Plan the analysis approach based on query intent"""
    print("📋 Planning analysis approach...")
    
    input_msg = f"""
    Original Query: {original_query}
    
    Interpreted Intent:
    - Commodity: {query_intent.commodity}
    - Analysis Type: {query_intent.query_type.value}
    - Focus Areas: {query_intent.specific_focus}
    - Time Horizon: {query_intent.time_horizon}
    - Additional Commodities: {query_intent.additional_commodities}
    - Confidence: {query_intent.confidence_score}
    
    Create an optimal analysis plan to address this query.
    """
    
    result = await Runner.run(analysis_coordinator_agent, input_msg)
    print(f"📊 Analysis plan created for {result.final_output.primary_commodity}")
    return result.final_output

async def execute_targeted_analysis(analysis_scope: AnalysisScope) -> dict:
    """
    NEW FUNCTION: Execute analysis based on the planned scope
    KEY INTEGRATION POINT: This function calls the EXISTING analyze_commodity_geopolitics() function
    """
    print(f"🔬 Executing {analysis_scope.analysis_type.value} analysis...")
    
    results = {}
    
    # Execute primary commodity analysis using EXISTING FUNCTION
    print(f"📊 Running primary analysis for {analysis_scope.primary_commodity}...")
    primary_analysis = await analyze_commodity_geopolitics(analysis_scope.primary_commodity)
    results['primary_analysis'] = primary_analysis
    
    # Execute comparative analysis if needed
    if analysis_scope.comparison_commodities:
        print(f"📈 Running comparative analysis for {len(analysis_scope.comparison_commodities)} additional commodities...")
        comparison_tasks = [
            asyncio.create_task(analyze_commodity_geopolitics(commodity))
            for commodity in analysis_scope.comparison_commodities
        ]
        comparison_results = await asyncio.gather(*comparison_tasks, return_exceptions=True)
        
        results['comparative_analysis'] = {
            commodity: result 
            for commodity, result in zip(analysis_scope.comparison_commodities, comparison_results)
            if not isinstance(result, Exception)
        }
    
    print("✅ Targeted analysis completed")
    return results

async def generate_conversational_response(user_query: str, query_intent: QueryIntent, analysis_results: dict) -> str:
    """NEW FUNCTION: Generate conversational response to user query"""
    print("💬 Generating conversational response...")
    
    # Prepare analysis summary for conversational agent
    primary_analysis = analysis_results['primary_analysis']
    
    input_msg = f"""
    User's Original Question: {user_query}
    
    Query Intent Analysis:
    - Commodity: {query_intent.commodity}
    - Analysis Type: {query_intent.query_type.value}
    - Focus Areas: {', '.join(query_intent.specific_focus)}
    - Time Horizon: {query_intent.time_horizon}
    
    Analysis Results:
    Executive Summary: {primary_analysis.executive_summary}
    Current Price Trend: {primary_analysis.current_price_trend}
    Key Geopolitical Factors: {primary_analysis.key_geopolitical_factors}
    Price Drivers: {primary_analysis.price_drivers}
    Risk Assessment: {primary_analysis.risk_assessment}
    Market Outlook: {primary_analysis.market_outlook}
    
    Please provide a conversational response that directly answers the user's question using this analysis.
    """
    
    result = await Runner.run(conversational_response_agent, input_msg)
    print("✅ Conversational response generated")
    return result.final_output

# ==================== MAIN INTERFACE FUNCTIONS (New) ====================

async def chat_with_commodity_analyst(user_query: str) -> dict:
    """
    NEW MAIN INTERFACE: Natural language query interface
    This is the primary entry point for natural language queries
    """
    
    try:
        # Process the natural language query
        response = await process_natural_language_query(user_query)
        
        return {
            "success": True,
            "user_query": user_query,
            "answer": response["conversational_response"],
            "confidence": response["confidence_score"],
            "analysis_details": {
                "commodity": response["query_intent"].commodity,
                "analysis_type": response["query_intent"].query_type.value,
                "technical_report": response["technical_analysis"]["primary_analysis"]
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "user_query": user_query,
            "error": str(e),
            "answer": f"I apologize, but I encountered an error processing your query: {str(e)}. Please try rephrasing your question."
        }

async def interactive_commodity_chat():
    """NEW FUNCTION: Interactive chat interface for commodity analysis"""
    
    print("🎯 INTERACTIVE COMMODITY ANALYST")
    print("Ask me anything about commodity prices, geopolitical impacts, market trends!")
    print("Type 'quit' to exit, 'help' for examples")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\n💬 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the Commodity Analyst!")
                break
                
            elif user_input.lower() in ['help', 'h']:
                print("\n📋 Example queries:")
                print("• What will oil prices do given current tensions?")
                print("• How are sanctions affecting gold markets?")
                print("• Compare copper vs aluminum in current market")
                print("• What are risks to wheat supply this year?")
                print("• Will natural gas prices rise with winter?")
                continue
                
            elif not user_input:
                print("Please enter a question about commodities.")
                continue
            
            print("\n🔍 Analyzing your question...")
            
            response = await chat_with_commodity_analyst(user_input)
            
            if response["success"]:
                print(f"\n🤖 {response['answer']}")
                print(f"\n📊 Analysis: {response['analysis_details']['commodity']} | {response['analysis_details']['analysis_type']}")
                print(f"🎯 Confidence: {response['confidence']:.2f}")
            else:
                print(f"\n❌ {response['answer']}")
                
        except KeyboardInterrupt:
            print("\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")

# ==================== UNIFIED INTERFACE CLASS (New) ====================

class CommodityAnalyst:
    """
    NEW UNIFIED INTERFACE: Combines both programmatic and natural language commodity analysis
    This class provides a clean interface to both the original and new functionality
    """
    
    def __init__(self):
        self.system_ready = True
        print("🎯 Commodity Analyst System Initialized")
        print("✅ Geopolitical analysis agents ready")
        print("✅ Natural language processing ready")
    
    async def analyze(self, commodity: str) -> CommodityAnalysisReport:
        """
        EXISTING FUNCTIONALITY: Direct commodity analysis
        Calls the original analyze_commodity_geopolitics() function
        """
        return await analyze_commodity_geopolitics(commodity)
    
    async def chat(self, user_query: str) -> dict:
        """
        NEW FUNCTIONALITY: Natural language query interface
        """
        return await chat_with_commodity_analyst(user_query)
    
    async def interactive_session(self):
        """NEW FUNCTIONALITY: Start interactive chat session"""
        await interactive_commodity_chat()
    
    async def batch_analysis(self, commodities: list) -> dict:
        """
        ENHANCED FUNCTIONALITY: Analyze multiple commodities
        Uses the original analyze_commodity_geopolitics() function for each commodity
        """
        results = {}
        for commodity in commodities:
            try:
                results[commodity] = await self.analyze(commodity)
                print(f"✅ Completed analysis for {commodity}")
            except Exception as e:
                results[commodity] = f"Error: {str(e)}"
                print(f"❌ Failed analysis for {commodity}: {e}")
        return results

# ==================== PRODUCTION INTERFACE (New) ====================

async def production_interface():
    """NEW FUNCTION: Production-ready interface with error handling"""
    
    analyst = CommodityAnalyst()
    
    while True:
        try:
            print("\n🎯 COMMODITY ANALYST - Choose Mode:")
            print("1. Natural Language Query")
            print("2. Direct Commodity Analysis") 
            print("3. Batch Analysis")
            print("4. Interactive Chat")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                query = input("💬 Your question: ")
                response = await analyst.chat(query)
                print(f"\n🤖 {response.get('answer', 'Error processing query')}")
                
            elif choice == "2":
                commodity = input("📊 Commodity name: ")
                report = await analyst.analyze(commodity)
                print(f"\n📈 {report.executive_summary}")
                
            elif choice == "3":
                commodities_input = input("📋 Commodities (comma-separated): ")
                commodities = [c.strip() for c in commodities_input.split(",")]
                results = await analyst.batch_analysis(commodities)
                for commodity, result in results.items():
                    if isinstance(result, CommodityAnalysisReport):
                        print(f"✅ {commodity}: {result.executive_summary}")
                    else:
                        print(f"❌ {commodity}: {result}")
                        
            elif choice == "4":
                await analyst.interactive_session()
                
            elif choice == "5":
                print("👋 Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# ==================== DEMO AND TESTING FUNCTIONS (New) ====================

async def demo_integrated_system():
    """NEW FUNCTION: Demonstrate both analysis modes"""
    
    analyst = CommodityAnalyst()
    
    print("\n" + "="*60)
    print("🎯 DEMO: ORIGINAL FUNCTIONALITY (Unchanged)")
    print("="*60)
    
    # Test original functionality
    print("Testing original analyze_commodity_geopolitics() function...")
    oil_report = await analyst.analyze("oil")
    print(f"✅ Original Function Works: {oil_report.executive_summary[:100]}...")
    
    print("\n" + "="*60)  
    print("🎯 DEMO: NEW NATURAL LANGUAGE FUNCTIONALITY")
    print("="*60)
    
    # Test natural language queries
    queries = [
        "What will oil prices do with current tensions?",
        "How are sanctions affecting gold?",
        "Compare copper vs aluminum markets"
    ]
    
    for query in queries:
        print(f"\n💬 Query: {query}")
        response = await analyst.chat(query)
        if response["success"]:
            print(f"🤖 Answer: {response['answer'][:150]}...")
            print(f"📊 Commodity: {response['analysis_details']['commodity']}")
        else:
            print(f"❌ Error: {response['error']}")

async def test_integration():
    """NEW FUNCTION: Test integration between old and new systems"""
    
    print("🧪 INTEGRATION TEST")
    print("="*50)
    
    # Test 1: Original function still works
    print("Test 1: Original function...")
    try:
        report = await analyze_commodity_geopolitics("gold")
        print(f"✅ Original function works: {bool(report.executive_summary)}")
    except Exception as e:
        print(f"❌ Original function failed: {e}")
    
    # Test 2: Natural language processing works
    print("\nTest 2: Natural language processing...")
    try:
        response = await chat_with_commodity_analyst("What will gold prices do?")
        print(f"✅ Natural language works: {response['success']}")
    except Exception as e:
        print(f"❌ Natural language failed: {e}")
    
    # Test 3: Unified interface works
    print("\nTest 3: Unified interface...")
    try:
        analyst = CommodityAnalyst()
        direct_result = await analyst.analyze("copper")
        chat_result = await analyst.chat("How is copper performing?")
        print(f"✅ Unified interface works: {isinstance(direct_result, CommodityAnalysisReport) and chat_result['success']}")
    except Exception as e:
        print(f"❌ Unified interface failed: {e}")

# ==================== MAIN EXECUTION (Updated) ====================

if __name__ == "__main__":
    # Choose your execution mode:
    
    # Option 1: Run production interface
    print("🚀 Starting Production Interface...")
    asyncio.run(production_interface())
    
    # Option 2: Run demo (uncomment to use)
    # print("🚀 Running Integration Demo...")
    # asyncio.run(demo_integrated_system())
    
    # Option 3: Run tests (uncomment to use)
    # print("🧪 Running Integration Tests...")
    # asyncio.run(test_integration())
    
    # Option 4: Run interactive chat directly (uncomment to use)
    # print("🚀 Starting Interactive Chat...")
    # asyncio.run(interactive_commodity_chat())

# ==================== INTEGRATION SUMMARY ====================
"""
INTEGRATION SUMMARY:

✅ UNCHANGED FUNCTIONS (Work exactly as before):
- analyze_commodity_geopolitics() - Main analysis function
- All orchestration functions (plan_geopolitical_searches, execute_geopolitical_research, etc.)
- All original agents (geopolitical_planner_agent, commodity_research_agent, etc.)
- All original data models (CommodityAnalysisReport, GeopoliticalSearchPlan, etc.)

➕ NEW FUNCTIONS (Added for natural language):
- chat_with_commodity_analyst() - Main natural language interface
- process_natural_language_query() - Natural language processing pipeline
- interpret_user_query() - Query interpretation
- CommodityAnalyst class - Unified interface

🔄 INTEGRATION POINTS:
- execute_targeted_analysis() calls analyze_commodity_geopolitics() internally
- Natural language layer wraps existing functionality
- All existing APIs remain unchanged
- New conversational interface added on top

🎯 USAGE:
# Original usage (unchanged):
report = await analyze_commodity_geopolitics("oil")

# New natural language usage:
response = await chat_with_commodity_analyst("What will oil prices do?")

# Unified interface:
analyst = CommodityAnalyst()
report = await analyst.analyze("oil")  # Original functionality
response = await analyst.chat("What will oil prices do?")  # New functionality
"""