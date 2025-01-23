import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from openai import OpenAI
from typing import List, Tuple, Dict, Optional
import re
from scipy import stats
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import json
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
OPENAI_CLIENT = None
OPENAI_CONFIG_FILE = 'auth.yaml'

class DataProcessor:
    def __init__(self):
        self.data = None
        self.equipment_list = []
        self.timestamp_col = None
        
    def load_data(self, file_path: str) -> Dict:
        try:
            logger.info(f"Loading data from: {file_path}")
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded data columns: {self.data.columns}")
            
            self.timestamp_col = "Date/Time"
            # Fix datetime parsing by removing extra spaces
            self.data[self.timestamp_col] = pd.to_datetime(self.data[self.timestamp_col].str.strip(), format='%m/%d %H:%M:%S')
            
            # Get equipment columns (excluding timestamp)
            self.equipment_list = [col for col in self.data.columns if col != self.timestamp_col]
            logger.info(f"Equipment list: {self.equipment_list}")
            
            # Get min and max dates
            dates = self.data[self.timestamp_col].dt.date.unique()
            min_date = min(dates).strftime('%Y-%m-%d')
            max_date = max(dates).strftime('%Y-%m-%d')
            
            summary = {
                'equipment_list': self.equipment_list,
                'min_date': min_date,
                'max_date': max_date,
                'total_records': len(self.data)
            }
            logger.info(f"Generated summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            return {'error': str(e), 'equipment_list': [], 'min_date': None, 'max_date': None}

    def clean_data(self) -> None:
        if self.data is None:
            return
            
        # Set the timestamp as index for proper time-based interpolation
        self.data.set_index(self.timestamp_col, inplace=True)
        
        for col in self.equipment_list:
            # Handle missing values with time-based interpolation
            self.data[col] = self.data[col].interpolate(method='time')
            
            # Remove outliers using IQR method
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            self.data[col] = self.data[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
        # Reset index to keep timestamp as a column
        self.data.reset_index(inplace=True)

    def get_filtered_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        if not start_date or not end_date:
            logger.warning("Missing date parameters in get_filtered_data")
            return self.data
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Create a copy to avoid modifying original data
            filtered_df = self.data.copy()
            
            # Convert timestamps to dates for comparison
            mask = (pd.to_datetime(filtered_df[self.timestamp_col]).dt.date >= start_date.date()) & \
                   (pd.to_datetime(filtered_df[self.timestamp_col]).dt.date <= end_date.date())
            
            return filtered_df[mask]
        except Exception as e:
            logger.error(f"Error in get_filtered_data: {str(e)}")
            return self.data

    def get_yearly_data(self, end_date: str) -> pd.DataFrame:
        if not end_date:
            return self.data
        try:
            end_date = pd.to_datetime(end_date)
            start_date = end_date - timedelta(days=365)
            return self.get_filtered_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        except Exception as e:
            logger.error(f"Error in get_yearly_data: {str(e)}")
            return self.data

class AnalyticsEngine:
    @staticmethod
    def analyze_daily_seasonality(data: pd.Series) -> Dict:
        try:
            decomposition = seasonal_decompose(data, period=24)
            return {
                'seasonal_pattern': 'Daily' if decomposition.seasonal.std() > 0.1 else 'No clear pattern',
                'trend': 'Increasing' if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0] else 'Decreasing',
                'peak_hours': data.groupby(data.index.hour).mean().nlargest(3).index.tolist()
            }
        except Exception as e:
            logger.error(f"Error in daily seasonality analysis: {str(e)}")
            return {
                'seasonal_pattern': 'Unable to determine',
                'trend': 'Unable to determine',
                'peak_hours': []
            }

    @staticmethod
    def analyze_yearly_seasonality(data: pd.Series) -> Dict:
        try:
            daily_data = data.resample('D').mean()
            decomposition = seasonal_decompose(daily_data, period=7)
            monthly_avg = data.groupby(data.index.month).mean()
            peak_months = monthly_avg.nlargest(3).index.tolist()
            
            return {
                'weekly_pattern': 'Present' if decomposition.seasonal.std() > 0.1 else 'No clear pattern',
                'yearly_trend': 'Increasing' if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0] else 'Decreasing',
                'peak_months': peak_months,
                'monthly_variation': monthly_avg.std() / monthly_avg.mean()
            }
        except Exception as e:
            logger.error(f"Error in yearly seasonality analysis: {str(e)}")
            return {
                'weekly_pattern': 'Unable to determine',
                'yearly_trend': 'Unable to determine',
                'peak_months': [],
                'monthly_variation': 0
            }

    @staticmethod
    def calculate_downtime(data: pd.Series) -> int:
        try:
            threshold = data.mean() * 0.1
            return (data <= threshold).sum()
        except Exception as e:
            logger.error(f"Error calculating downtime: {str(e)}")
            return 0

    @staticmethod
    def calculate_basic_metrics(data: pd.Series) -> Dict:
        try:
            return {
                'mean': data.mean(),
                'max': data.max(),
                'min': data.min(),
                'std': data.std(),
                'total_consumption': data.sum()
            }
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            return {
                'mean': 0,
                'max': 0,
                'min': 0,
                'std': 0,
                'total_consumption': 0
            }

class Visualizer:
    @staticmethod
    def create_timeline_plot(data: pd.DataFrame, selected_equipment: List[str], 
                           timestamp_col: str) -> go.Figure:
        try:
            # Create a copy of the data to modify timestamps
            plot_data = data.copy()
            # Set year to 2024 for display
            plot_data[timestamp_col] = plot_data[timestamp_col].apply(
                lambda x: x.replace(year=2024)
            )
            
            fig = go.Figure()
            for equipment in selected_equipment:
                fig.add_trace(go.Scatter(
                    x=plot_data[timestamp_col],
                    y=plot_data[equipment],
                    name=equipment,
                    mode='lines'
                ))
            fig.update_layout(
                title='Power Consumption Timeline',
                xaxis_title='Time',
                yaxis_title='Power Consumption (kW)',
                showlegend=True,
                width=1200,  # Make plot wider
                height=500,  # Adjust height
                margin=dict(l=50, r=50, t=50, b=50)
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating timeline plot: {str(e)}")
            return None

    @staticmethod
    def create_energy_ranking_plot(data: pd.DataFrame, 
                                 selected_equipment: List[str]) -> go.Figure:
        try:
            totals = {equip: data[equip].sum() for equip in selected_equipment}
            sorted_equipment = sorted(totals.items(), key=lambda x: x[1], reverse=True)
            
            fig = go.Figure([go.Bar(
                x=[item[0] for item in sorted_equipment],
                y=[item[1] for item in sorted_equipment]
            )])
            fig.update_layout(
                title='Total Energy Consumption Ranking',
                xaxis_title='Equipment',
                yaxis_title='Total Consumption',
                showlegend=False
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating ranking plot: {str(e)}")
            return None

class QueryProcessor:
    @staticmethod
    def extract_info_from_prompt(prompt: str) -> Tuple[List[str], Tuple[str, str]]:
        if OPENAI_CLIENT is None:
            logger.error("OpenAI client not initialized")
            return [], (None, None)
            
        system_prompt = """
        Extract the following information from the user query:
        1. Equipment names mentioned (if any)
        2. Time period mentioned (if any)
        Format the response as JSON with keys: 'equipment' (list) and 'period' (dict with 'start' and 'end' keys)
        """
        
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            result = eval(response.choices[0].message.content)
            return result['equipment'], (result['period']['start'], result['period']['end'])
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            return [], (None, None)

    @staticmethod
    def generate_analysis_response(prompt: str, analysis_data: Dict) -> str:
        if OPENAI_CLIENT is None:
            return "OpenAI client not initialized"
            
        system_prompt = """
        You are an energy consumption analysis expert. Based on the provided analysis data 
        and the original user query:
        1. Answer the user's question directly
        2. Highlight key insights from the analysis
        3. Point out concerning patterns or anomalies
        4. Suggest energy optimization opportunities
        Keep the response clear and concise while being informative.
        """
        
        try:
            # Convert numpy/pandas types to native Python types for JSON serialization
            def convert_to_native_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_to_native_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native_types(item) for item in obj]
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return convert_to_native_types(obj.tolist())
                return obj
                
            serializable_data = convert_to_native_types(analysis_data)
            context = json.dumps(serializable_data, indent=2)
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original Query: {prompt}\n\nAnalysis Data: {context}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating analysis response: {str(e)}")
            return f"Error generating analysis response: {str(e)}"

class PowerAnalysisDriver:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.analytics = AnalyticsEngine()
        self.visualizer = Visualizer()
        self.query_processor = QueryProcessor()

    def process_request(self, query: str, file_path: str, selected_equipment: List[str], 
                       start_date: str, end_date: str) -> Tuple[go.Figure, go.Figure, str]:
        try:
            if not selected_equipment:
                logger.warning("No equipment selected")
                return None, None, "Please select at least one equipment"

            if file_path:
                self.data_processor.load_data(file_path)
                self.data_processor.clean_data()

            if not (start_date and end_date):
                _, extracted_period = self.query_processor.extract_info_from_prompt(query)
                start_date = start_date or extracted_period[0]
                end_date = end_date or extracted_period[1]

            filtered_data = self.data_processor.get_filtered_data(start_date, end_date)
            yearly_data = self.data_processor.get_yearly_data(end_date)

            analysis_data = {
                'selected_period': {'start': start_date, 'end': end_date},
                'equipment_analysis': {}
            }

            for equipment in selected_equipment:
                equipment_data = filtered_data[equipment]
                yearly_equipment_data = yearly_data[equipment]
                
                equipment_data.index = filtered_data[self.data_processor.timestamp_col]
                yearly_equipment_data.index = yearly_data[self.data_processor.timestamp_col]

                analysis_data['equipment_analysis'][equipment] = {
                    'basic_metrics': self.analytics.calculate_basic_metrics(equipment_data),
                    'daily_patterns': self.analytics.analyze_daily_seasonality(equipment_data),
                    'yearly_patterns': self.analytics.analyze_yearly_seasonality(yearly_equipment_data),
                    'downtime_hours': self.analytics.calculate_downtime(equipment_data)
                }

            timeline = self.visualizer.create_timeline_plot(
                filtered_data, selected_equipment, self.data_processor.timestamp_col)
            ranking = self.visualizer.create_energy_ranking_plot(filtered_data, selected_equipment)

            ai_response = self.query_processor.generate_analysis_response(query, analysis_data)

            summary_parts = [f"AI Analysis:\n{ai_response}\n\nDetailed Equipment Analysis:"]
            
            for equipment, analysis in analysis_data['equipment_analysis'].items():
                summary_parts.append(f"""
                Equipment: {equipment}
                - Consumption Metrics:
                  * Average: {analysis['basic_metrics']['mean']:.2f}
                  * Maximum: {analysis['basic_metrics']['max']:.2f}
                  * Total: {analysis['basic_metrics']['total_consumption']:.2f}
                - Daily Patterns:
                  * Pattern: {analysis['daily_patterns']['seasonal_pattern']}
                  * Peak Hours: {analysis['daily_patterns']['peak_hours']}
                - Yearly Patterns:
                  * Weekly Pattern: {analysis['yearly_patterns']['weekly_pattern']}
                  * Trend: {analysis['yearly_patterns']['yearly_trend']}
                  * Peak Months: {analysis['yearly_patterns']['peak_months']}
                - Downtime: {analysis['downtime_hours']} hours
                """)

            return timeline, ranking, "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return None, None, f"Error in analysis: {str(e)}"

def create_gradio_interface(driver: PowerAnalysisDriver):
    def process_query(query: str, file, selected_equipment, start_date, end_date):
        try:
            if file is not None:
                return driver.process_request(query, file.name, selected_equipment, start_date, end_date)
            return None, None, "Please upload a file first"
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return None, None, f"Error processing query: {str(e)}"

    def update_interface_elements(file):
        try:
            if file is not None:
                summary = driver.data_processor.load_data(file.name)
                equipment_list = summary.get('equipment_list', [])
                min_date = summary.get('min_date', '')
                max_date = summary.get('max_date', '')
                
                logger.info(f"Update values - Equipment: {equipment_list}, Dates: {min_date} to {max_date}")
                
                return gr.Dropdown(choices=equipment_list, multiselect=True), min_date, max_date
            return gr.Dropdown(choices=[], multiselect=True), "", ""
        except Exception as e:
            logger.error(f"Error in update_interface_elements: {str(e)}")
            return gr.Dropdown(choices=[], multiselect=True), "", ""

    def clear_outputs():
        return None, None, "", gr.Dropdown(choices=[], multiselect=True), "", ""

    with gr.Blocks(title="Hospital Power Consumption Analysis") as interface:
        with gr.Column():
            gr.Markdown("# Hospital Power Consumption Analysis System")
            
            file_input = gr.File(
                label="Upload Power Consumption Data (CSV)",
                file_types=[".csv"],
                type="filepath"
            )
            
            equipment_dropdown = gr.Dropdown(
                choices=[], 
                multiselect=True,
                label="Select Equipment",
                interactive=True
            )
            
            start_date = gr.Textbox(
                label="Start Date (YYYY-MM-DD)",
                placeholder="YYYY-MM-DD",
                interactive=True
            )
            
            end_date = gr.Textbox(
                label="End Date (YYYY-MM-DD)",
                placeholder="YYYY-MM-DD",
                interactive=True
            )
            
            query_input = gr.Textbox(
                lines=2,
                placeholder="Enter your query about power consumption...",
                label="Query"
            )
            
            with gr.Row():
                analyze_button = gr.Button("Analyze", variant="primary")
                clear_button = gr.Button("Clear")
            
            # Timeline plot takes full width
            out_timeline = gr.Plot(label="Power Consumption Timeline")
            
            # Create two columns for summary and ranking
            with gr.Row():
                with gr.Column(scale=2):  # Left column for summary (wider)
                    out_summary = gr.Textbox(
                        label="Analysis Summary and Insights",
                        lines=15,
                        interactive=False
                    )
                with gr.Column(scale=1):  # Right column for ranking plot (narrower)
                    out_ranking = gr.Plot(label="Equipment Energy Ranking")

        # Event handlers
        file_input.change(
            fn=update_interface_elements,
            inputs=[file_input],
            outputs=[
                equipment_dropdown,
                start_date,
                end_date
            ]
        )
        
        analyze_button.click(
            fn=process_query,
            inputs=[
                query_input,
                file_input,
                equipment_dropdown,
                start_date,
                end_date
            ],
            outputs=[
                out_timeline,
                out_ranking,
                out_summary
            ]
        )
        
        clear_button.click(
            fn=clear_outputs,
            inputs=[],
            outputs=[
                out_timeline,
                out_ranking,
                out_summary,
                equipment_dropdown,
                start_date,
                end_date
            ]
        )
    
    return interface

def main():
    """Main function to initialize and launch the application."""
    global OPENAI_CLIENT
    
    try:
        # Load API key from config file
        with open(OPENAI_CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        apikey = config['openai']['access_key']
        
        # Initialize OpenAI client globally
        OPENAI_CLIENT = OpenAI(api_key=apikey)
        logger.info("OpenAI client initialized successfully")
        
        # Initialize the driver
        driver = PowerAnalysisDriver()
        
        # Create and launch the interface
        interface = create_gradio_interface(driver)
        interface.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860
        )
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()