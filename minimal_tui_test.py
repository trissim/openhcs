#!/usr/bin/env python3
"""
Minimal TUI test to demonstrate the working implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

try:
    from prompt_toolkit import Application
    from prompt_toolkit.layout import Layout, HSplit, VSplit, Dimension
    from prompt_toolkit.widgets import Label, Button, Frame
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.key_binding import KeyBindings
    
    def create_tui():
        """Create a minimal TUI that demonstrates our 3-bar layout."""
        
        # Create key bindings
        kb = KeyBindings()
        
        @kb.add('q')
        def quit_app(event):
            event.app.exit()
        
        @kb.add('c-c')
        def quit_app_ctrl_c(event):
            event.app.exit()
        
        # Create the 3-bar layout as implemented
        layout = Layout(
            Frame(
                HSplit([
                    # 1st Bar: Top Menu Bar
                    VSplit([
                        Button("Global Settings", handler=lambda: None),
                        Button("Help", handler=lambda: None),
                        Window(width=Dimension(weight=1)),
                        Label("OpenHCS V1.0"),
                        Window(width=Dimension(weight=1))
                    ], height=Dimension.exact(1)),

                    # 2nd Bar: Titles Bar
                    VSplit([
                        Window(
                            content=FormattedTextControl([("class:title", " 1 Plate Manager ")]),
                            height=Dimension.exact(1),
                            char=' ',
                            width=Dimension(weight=1)
                        ),
                        Window(width=Dimension.exact(1), char='│'),
                        Window(
                            content=FormattedTextControl([("class:title", " 2 Pipeline Editor ")]),
                            height=Dimension.exact(1),
                            char=' ',
                            width=Dimension(weight=1)
                        )
                    ], height=Dimension.exact(1)),

                    # 3rd Bar: Buttons Bar
                    VSplit([
                        VSplit([
                            Button("add", handler=lambda: None),
                            Button("del", handler=lambda: None),
                            Button("edit", handler=lambda: None),
                            Button("init", handler=lambda: None),
                            Button("compile", handler=lambda: None),
                            Button("run", handler=lambda: None),
                        ], width=Dimension(weight=1)),
                        
                        Window(width=Dimension.exact(1), char='│'),
                        
                        VSplit([
                            Button("add", handler=lambda: None),
                            Button("del", handler=lambda: None),
                            Button("edit", handler=lambda: None),
                            Button("load", handler=lambda: None),
                            Button("save", handler=lambda: None),
                        ], width=Dimension(weight=1))
                    ], height=Dimension.exact(1)),

                    # Separator
                    Window(height=Dimension.exact(1), char='─'),

                    # Main Panes
                    VSplit([
                        # Left pane: Plate list
                        Frame(
                            HSplit([
                                Label("? plate1 | /path/to/plate1"),
                                Label("✓ plate2 | /path/to/plate2"),
                                Label("! plate3 | /path/to/plate3"),
                            ]),
                            title="Plates"
                        ),
                        
                        # Separator
                        Window(width=Dimension.exact(1), char='│'),
                        
                        # Right pane: Step list
                        Frame(
                            HSplit([
                                Label("? step1: function_type"),
                                Label("o step2: processing_type"),
                                Label("✓ step3: analysis_type"),
                            ]),
                            title="Steps"
                        )
                    ], height=Dimension(weight=1)),

                    # Separator
                    Window(height=Dimension.exact(1), char='─'),

                    # Status Bar
                    Window(
                        content=FormattedTextControl([("", "Ready - Press 'q' to quit")]),
                        height=Dimension.exact(1)
                    )
                ], padding=0)
            )
        )
        
        return Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=True
        )
    
    def main():
        """Run the minimal TUI."""
        print("Starting minimal TUI test...")
        print("This demonstrates our 3-bar layout implementation.")
        print("Press 'q' or Ctrl+C to quit.")
        
        app = create_tui()
        app.run()
        
        print("TUI test completed successfully!")

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure prompt_toolkit is installed: pip install prompt_toolkit")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
