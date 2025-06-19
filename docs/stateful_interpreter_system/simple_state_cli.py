import json
import os

class SimpleStateCLI:
    def __init__(self, state_file="cli_state.json"):
        self.state_file = state_file
        self.state = self.load_state()
    
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "objectives": {},
            "key_store": {},
            "metrics": {}
        }
    
    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def set_value(self, key, value):
        self.state['key_store'][key] = value
        self.save_state()
        return f"Set {key} = {value}"
    
    def get_value(self, key):
        return self.state['key_store'].get(key, "Not found")
    
    def set_objective(self, objective, value):
        self.state['objectives'][objective] = value
        self.save_state()
        return f"Objective set: {objective} = {value}"
    
    def get_progress(self):
        report = "Progress Report:\n"
        for objective, target in self.state['objectives'].items():
            current = self.state['metrics'].get(objective, 0)
            progress = min(100, (current / target) * 100) if target != 0 else 0
            report += f"- {objective}: {current}/{target} ({progress:.1f}%)\n"
        return report

# Example usage:
# cli = SimpleStateCLI()
# cli.set_objective("docs_restructuring", 100)
# print(cli.get_progress())