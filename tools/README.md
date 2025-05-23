# Generic Code Analysis Tools

**🎯 Purpose**: Automated codebase analysis and architectural insights for any Python project.

## For AI Agents: Quick Discovery Guide

If you're an AI agent helping with codebase analysis, here's your workflow:

### 1. 📖 Read This Directory
You've found a complete code analysis toolkit. Here's what's available:

### 2. 🔍 Available Analysis Capabilities

- **📊 Project Snapshot** - High-level overview of entire codebase
- **🔬 Detailed Analysis** - Deep dive into specific files/modules  
- **🕸️ Dependency Mapping** - Understand module relationships
- **⚡ Quick Analysis** - One-command comprehensive analysis

### 3. 🤖 Your Next Steps as an AI Agent

1. **Ask the human**: "Which directory should I analyze?" 
2. **Explain what you understand**: "I can provide [list capabilities]"
3. **Offer specific help**: "Would you like me to [specific analysis]?"

### 4. 🛠️ Tool Commands (Generic)

```bash
# Get high-level overview of any Python project
python tools/code_analysis/code_analyzer_cli.py snapshot --target <directory>

# Analyze specific files in detail  
python tools/code_analysis/code_analyzer_cli.py matrix <file1> <file2> ...

# Map dependencies for any directory
python tools/code_analysis/code_analyzer_cli.py dependencies <directory>

# Quick comprehensive analysis
./tools/quick_start.sh <target_directory>
```

### 5. 📁 Output Structure

All reports go to: `reports/code_analysis/`
- `*_snapshot.csv` - Quantitative project overview
- `*_matrix.md` - Detailed code analysis (human-readable)
- `*_dependencies.md` - Module relationship mapping

### 6. 🎯 Common AI Agent Conversation Flow

**Human**: "Help me understand this codebase"
**You**: "I see you have code analysis tools available. Which directory should I analyze? I can provide:
- Complete project overview
- Detailed module analysis  
- Dependency mapping
- Architectural insights"

**Human**: "Analyze the src/ directory"
**You**: "Running analysis on src/... [execute commands] ... Here's what I found: [insights from reports]"

### 7. 💡 Agent Best Practices

- Always ask which directory to analyze (don't assume)
- Run analysis BEFORE making architectural suggestions
- Use quantitative data from CSV files for metrics
- Use Markdown reports for qualitative insights
- Check dependencies before suggesting changes
- Offer specific, actionable recommendations

### 8. 🔧 Tool Details

See `tools/code_analysis/README.md` for technical details and advanced usage.

---

**🚀 Ready to help? Ask the human which directory to analyze!**