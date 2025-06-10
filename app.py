"""
Streamlit UI for comparing OpenAI models.

Usage:
    streamlit run app.py
"""
import asyncio
import streamlit as st
from test import ask_one_async, AsyncOpenAI, PRICING_PER_MILLION
import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="OpenAI Model Comparison",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better number formatting
st.markdown("""
<style>
    .cost-value {
        font-family: monospace;
        font-size: 1.1em;
    }
    .metric-value {
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 OpenAI Model Comparison")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    available_models = list(PRICING_PER_MILLION.keys())
    model1 = st.selectbox("Model 1", available_models, index=0)
    model2 = st.selectbox("Model 2", available_models, index=2)
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API Key")
    
    # Pricing information
    st.header("Pricing (per million tokens)")
    for model, prices in PRICING_PER_MILLION.items():
        st.text(f"{model}:")
        st.text(f"  Input: ${prices['input']:.3f}")
        st.text(f"  Output: ${prices['output']:.3f}")

# Main content
prompt = st.text_area("Enter your prompt:", height=150)

def format_cost(cost: float) -> str:
    """Format cost with appropriate precision."""
    if cost < 0.000001:
        return f"${cost:.8f}"
    elif cost < 0.0001:
        return f"${cost:.6f}"
    elif cost < 0.01:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"

if st.button("Compare Models") and api_key and prompt:
    if not prompt.strip():
        st.error("Please enter a prompt")
    else:
        with st.spinner("Running models..."):
            # Create async client
            client = AsyncOpenAI(api_key=api_key)
            
            # Run models concurrently
            async def run_comparison():
                tasks = [
                    ask_one_async(prompt, model1, client),
                    ask_one_async(prompt, model2, client)
                ]
                return await asyncio.gather(*tasks)
            
            results = asyncio.run(run_comparison())
            
            # Save results to JSONL file
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "comparison_logs.jsonl"
            
            # Add comparison metadata
            comparison_record = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "model1": model1,
                "model2": model2,
                "results": results
            }
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(comparison_record, ensure_ascii=False) + "\n")
            
            st.success(f"Results saved to {log_file}")
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            for i, (col, result) in enumerate(zip([col1, col2], results)):
                with col:
                    st.subheader(f"Model: {result['model']}")
                    st.markdown("---")
                    st.markdown(result["response"])
                    st.markdown("---")
                    
                    # Create metrics container
                    metrics_container = st.container()
                    with metrics_container:
                        st.metric("Latency", f"{result['latency_ms']:.1f} ms")
                        st.metric("Prompt Tokens", f"{result['prompt_tokens']:,}")
                        st.metric("Completion Tokens", f"{result['completion_tokens']:,}")
                        st.metric("Total Tokens", f"{result['total_tokens']:,}")
                        st.markdown(f"**Cost:** <span class='cost-value'>{format_cost(result['cost_usd'])}</span>", unsafe_allow_html=True)
            
            # Add comparison chart
            st.subheader("Comparison Chart")
            
            # Prepare data for visualization
            metrics_data = {
                "Metric": ["Latency (ms)", "Prompt Tokens", "Completion Tokens", "Total Tokens"],
                model1: [
                    results[0]["latency_ms"],
                    results[0]["prompt_tokens"],
                    results[0]["completion_tokens"],
                    results[0]["total_tokens"]
                ],
                model2: [
                    results[1]["latency_ms"],
                    results[1]["prompt_tokens"],
                    results[1]["completion_tokens"],
                    results[1]["total_tokens"]
                ]
            }
            
            # Create separate charts for different metrics
            tab1, tab2 = st.tabs(["Token Usage", "Latency"])
            
            with tab1:
                # Token usage chart (log scale for better visualization)
                token_data = pd.DataFrame({
                    "Metric": ["Prompt", "Completion", "Total"],
                    model1: [
                        results[0]["prompt_tokens"],
                        results[0]["completion_tokens"],
                        results[0]["total_tokens"]
                    ],
                    model2: [
                        results[1]["prompt_tokens"],
                        results[1]["completion_tokens"],
                        results[1]["total_tokens"]
                    ]
                })
                st.bar_chart(token_data.set_index("Metric"), use_container_width=True)
            
            with tab2:
                # Latency chart
                latency_data = pd.DataFrame({
                    "Model": [model1, model2],
                    "Latency (ms)": [results[0]["latency_ms"], results[1]["latency_ms"]]
                })
                st.bar_chart(latency_data.set_index("Model"), use_container_width=True)
            
            # Cost comparison in a separate section
            st.subheader("Cost Comparison")
            cost_data = pd.DataFrame({
                "Model": [model1, model2],
                "Cost (USD)": [results[0]["cost_usd"], results[1]["cost_usd"]]
            })
            st.bar_chart(cost_data.set_index("Model"), use_container_width=True)
            
            # Display detailed cost breakdown
            st.markdown("### Cost Breakdown")
            cost_breakdown = pd.DataFrame({
                "Model": [model1, model2],
                "Prompt Cost": [
                    results[0]["prompt_tokens"] * PRICING_PER_MILLION[model1]["input"] / 1_000_000,
                    results[1]["prompt_tokens"] * PRICING_PER_MILLION[model2]["input"] / 1_000_000
                ],
                "Completion Cost": [
                    results[0]["completion_tokens"] * PRICING_PER_MILLION[model1]["output"] / 1_000_000,
                    results[1]["completion_tokens"] * PRICING_PER_MILLION[model2]["output"] / 1_000_000
                ],
                "Total Cost": [results[0]["cost_usd"], results[1]["cost_usd"]]
            })
            st.dataframe(cost_breakdown.style.format({
                "Prompt Cost": "${:.8f}",
                "Completion Cost": "${:.8f}",
                "Total Cost": "${:.8f}"
            })) 