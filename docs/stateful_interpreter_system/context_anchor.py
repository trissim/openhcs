import json
import os
from datetime import datetime

class ContextAnchor:
    CONFIG_PATH = "docs/ai_collaboration/anchor_config.json"
    DEFAULT_CONFIG = {
        "reminder_interval": 3,
        "current_goal": "Complete framework implementation",
        "memory_path": "docs/ai_collaboration/memory.json"
    }
    
    def __init__(self):
        self.message_count = 0
        self.config = self.load_config()
        self.last_reminder = 0
        
    def load_config(self):
        try:
            with open(self.CONFIG_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            with open(self.CONFIG_PATH, 'w') as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=2)
            return self.DEFAULT_CONFIG.copy()
    
    def update_config(self, key, value):
        self.config[key] = value
        with open(self.CONFIG_PATH, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def check_reminder(self, current_message_count):
        self.message_count = current_message_count
        if self.message_count - self.last_reminder >= self.config['reminder_interval']:
            self.last_reminder = self.message_count
            return self.generate_reminder()
        return None
    
    def generate_reminder(self):
        return f"""
CONTEXT ANCHOR [Message {self.message_count}]
Current Goal: {self.config['current_goal']}
Conversation Depth: {self.message_count} messages
Last Reset: Message {max(0, self.message_count - self.config['reminder_interval'])}
"""

# Example usage
if __name__ == "__main__":
    anchor = ContextAnchor()
    for i in range(1, 7):
        reminder = anchor.check_reminder(i)
        if reminder:
            print(reminder)