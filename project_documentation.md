# CostimAIze Project Documentation

## Project Overview
- **Project Name**: CostimAIze
- **Project ID**: CostimAIze-Project-2025
- **Objective**: Develop an AI-powered application that acts as a cautious and intelligent pricing engineer for project cost estimation, bid analysis, and historical price archiving.
- **Platform**: Streamlit Cloud
- **AI Model**: OpenAI (gpt-3.5-turbo)
- **Start Date**: April 2025

## Core Requirements
### General Requirements
- The application must act as an intelligent and cautious pricing engineer, capable of learning and evolving.
- All AI functions must use real, advanced intelligence (OpenAI) and not local or mock logic.
- All logic layers (including cautious pricing) must rely on OpenAI for decision-making.
- The application provides three main services:
  1. **Cost Estimation**: Generate a cost estimation report for a project based on a scope of work.
  2. **Bid Analysis**: Analyze bid prices based on a pricing table.
  3. **Historical Price Archiving**: Archive historical prices for 90 days.

### Cost Estimation Requirements
- No pre-programmed guidance for learning or development, except for the cautious pricing logic.
- Consider all direct and indirect costs, including global market conditions (inflation, material prices, interest rates, etc.).
- The AI determines influencing factors and their impact percentages independently.
- Estimation is based on executing the scope of work as a single, integrated project.
- The AI must continue working even if helper data is unavailable, with the option to apply reserve percentages if deemed necessary.
- Helper data (historical prices, global market conditions, global news) is provided, but the AI decides whether to use it, how to use it, and its impact.

### Cautious Pricing Logic
- Retain results for 90 days without changes.
- Run cost estimation 100 times using Monte Carlo simulation before finalizing the result.
- Ensure the estimated price is logical and aligns with the scope of work, project conditions, and global market conditions.
- Avoid randomness or hallucination in pricing.
- The AI itself decides the final result and the reasoning behind it, without local rules.

### Interface Requirements
- **Dashboard**: Main page displaying attractive, auto-updated statistics about the application's activities.
- Display the three main services (Cost Estimation, Bid Analysis, Historical Price Archiving) for user selection.
- Navigate to service-specific pages upon selection.
- **Cost Estimation Interface**:
  1. Window to upload scope of work files (Word, Excel, PDF, multiple files allowed).
  2. Real-time analysis of the scope, displaying contradictions in separate windows.
  3. Windows for user responses to contradictions.
  4. Final cost estimation report window.

## Current File Structure
