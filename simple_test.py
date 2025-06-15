#!/usr/bin/env python3
"""
Very simple test to see if Textual is working at all.
"""

print("Script starting...")

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Button, Static
    print("Textual imports successful")
except Exception as e:
    print(f"Import error: {e}")
    exit(1)

class SimpleApp(App):
    def compose(self) -> ComposeResult:
        print("SimpleApp.compose() called")
        yield Static("Hello World")
        yield Button("Test Button", id="test")
    
    def on_mount(self):
        print("SimpleApp mounted")
    
    def on_button_pressed(self, event):
        print(f"Button pressed: {event.button.id}")

if __name__ == "__main__":
    print("Creating app...")
    app = SimpleApp()
    print("Running app...")
    app.run()
    print("App finished")
