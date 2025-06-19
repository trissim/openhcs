import json
from datetime import datetime

class PersistentMemory:
    def __init__(self, interval=5, file_path="docs/ai_collaboration/memory.json"):
        self.message_count = 0
        self.interval = interval  # Visualize every X messages
        self.file_path = file_path
        self.history = []         # Stores message summaries
        self.visualizations = {}  # {message_count: diagram}
        
        # Load existing memory if available
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.message_count = data.get('message_count', 0)
                self.history = data.get('history', [])
                self.visualizations = data.get('visualizations', {})
        except FileNotFoundError:
            pass
        
    def add_message(self, user_input, ai_response):
        self.message_count += 1
        timestamp = datetime.now().isoformat()
        summary = {
            "timestamp": timestamp,
            "user": user_input[:500],
            "ai": ai_response[:500]
        }
        self.history.append(summary)
        
        if self.message_count % self.interval == 0:
            diagram = self.generate_diagram()
            self.visualizations[self.message_count] = diagram
            self.save_memory()
            return diagram
        return None
        
    def generate_diagram(self):
        """Creates visualization of conversation flow"""
        start = max(0, len(self.history) - self.interval)
        recent = self.history[start:]
        
        # Build Mermaid nodes
        nodes = []
        for i, msg in enumerate(recent, 1):
            node_id = f"M{start+i}"
            label = f"{start+i}: {msg['user'][:20]}..."
            nodes.append(f"{node_id}[{label}]")
        
        # Build connections
        connections = []
        for i in range(len(nodes)-1):
            connections.append(f"{nodes[i].split('[')[0]} --> {nodes[i+1].split('[')[0]}")
        
        return f"""```mermaid
flowchart LR
    subgraph Last {self.interval} Messages
        direction TB
        { '        '.join(nodes) }
        { '        '.join(connections) }
    end
```"""
    
    def save_memory(self):
        """Save memory state to file"""
        data = {
            "message_count": self.message_count,
            "history": self.history,
            "visualizations": self.visualizations
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)

# Example usage
if __name__ == "__main__":
    memory = PersistentMemory(interval=3, file_path="docs/ai_collaboration/memory.json")
    
    # Simulate conversation
    responses = [
        memory.add_message("Hello", "Hi there!"),
        memory.add_message("How are you?", "I'm functioning optimally"),
        memory.add_message("Tell me about AI", "AI is..."),  # This should trigger visualization
        memory.add_message("More info", "Here's more"),
        memory.add_message("Another question", "Another answer"),
        memory.add_message("Final input", "Final response")  # Trigger again
    ]
    
    # Print any generated visualizations
    for i, viz in enumerate(responses):
        if viz:
            print(f"Visualization at message {i+1}:")
            print(viz)