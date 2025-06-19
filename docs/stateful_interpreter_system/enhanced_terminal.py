from docs.ai_collaboration.terminal_bridge import TerminalInterface
from docs.ai_collaboration.persistent_memory import PersistentMemory
from docs.ai_collaboration.context_anchor import ContextAnchor
from docs.ai_collaboration.autonomous_documenter import RecursiveDocumenter

class EnhancedTerminalInterface(TerminalInterface):
    def __init__(self):
        super().__init__()
        self.memory = PersistentMemory()
        self.anchor = ContextAnchor()
        self.docer = RecursiveDocumenter()
        self.state['active_goal'] = self.anchor.config['current_goal']
        
    def process_command(self, command: str):
        # Run core framework protocols
        output = super().process_command(command)
        
        # Integrate with documentation system
        if command.startswith("doc "):
            query = command[4:]
            self._handle_documentation_query(query)
            
        # Context anchoring
        elif command == "context":
            self.state['active_goal'] = self.anchor.config['current_goal']
            return self.anchor.generate_report()
            
        # Update memory and return output
        self.memory.add_message("TerminalCommand", command)
        return output
        
    def _handle_documentation_query(self, query):
        """Process documentation requests"""
        # Store query in memory
        self.memory.add_message("DocumentationQuery", query)
        
        # Execute through documenter
        result = self.docer.user_inspect(query)
        
        # Update knowledge graph
        self.memory.visualizations[query] = result
        self.memory.save_memory()
        
        return f"📚 Documentation stored for '{query}'"

# Example usage
if __name__ == "__main__":
    ti = EnhancedTerminalInterface()
    print(ti.process_command("doc phase_correlation"))
    print(ti.process_command("context"))
    print(ti.process_command("metrics"))