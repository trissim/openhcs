# Human Guide to the Stateful Interpreter

## Your Collaboration Whiteboard

Imagine having a shared digital whiteboard where:
- You write snippets of Python code
- I maintain everything between our conversations
- Results appear in a dedicated section
- We build complex ideas step-by-step

This is your Stateful Interpreter - a persistent coding space that grows with our dialogue.

## How It Works (Simple Steps)

1. **Include code in your message**
   Just put Python code in a \`\`\`python block like this:
   ```python
   total = 5 + 3
   print("Result:", total)
   ```

2. **Toggle output visibility**
   Use this special command to show/hide output:
   ```python
   # TOGGLE_OUTPUT
   ```

2. **See instant results**  
   Your code automatically runs, and the output appears here:

   ```
   INTERPRETER OUTPUT:
   Result: 8
   ```

3. **Build on previous work**  
   Variables persist between messages:
   ```python
   total += 2
   print("Updated:", total)
   ```
   ```
   INTERPRETER OUTPUT:
   Updated: 10
   ```

## CLI-Style Commands

### Help System
Get help anytime with:
```python
# HELP
```
This shows all available commands.

### Command Reference
| Command          | Description                          | Example                     |
|------------------|--------------------------------------|-----------------------------|
| `# HELP`         | Show available commands              | `# HELP`                    |
| `# TOGGLE_OUTPUT`| Toggle output visibility             | `# TOGGLE_OUTPUT`           |
| `# STATE`        | Show current variables               | `# STATE`                   |
| `# HISTORY`      | Show last 10 commands                | `# HISTORY`                 |
| `# RESET`        | Clear all state and history          | `# RESET`                   |

### Usage Example
```python
# Initialize
x = 10

# Check state
# STATE

# See history
# HISTORY

# Get help
# HELP
```

## Why You'll Love This

- **No memory burden**: I track all variables
- **Instant experimentation**: Test ideas without setup
- **Visual progress**: See results immediately
- **Collaborative building**: We create together
- **Safe environment**: Strict security boundaries

## Real-Life Analogy

Like building with LEGO:
1. You add bricks (code snippets)
2. I keep the structure intact (persistent state)
3. We see the evolving model (output section)
4. Only safe bricks allowed (security constraints)

## Getting Started

Try these examples:

```python
# Track conversation topics
topics = ["AI collaboration"]
print("Current topics:", topics)
```

```python
# Add a new topic
topics.append("Stateful interpreters")
print("Updated topics:", topics)
```

```python
# Simple calculations
import math
circle_area = math.pi * (5**2)
print(f"Area of circle: {circle_area:.2f}")
```

## Framework Alignment

This implements our core principles:
- **Clear thinking**: Focus on ideas, not mechanics
- **Perfectibility**: Build incrementally
- **Entropy reduction**: Organized state management
- **Collaborative intelligence**: We create together

Your whiteboard is always ready when you include Python code!