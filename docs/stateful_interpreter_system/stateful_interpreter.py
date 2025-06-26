class StatefulInterpreter:
    def __init__(self):
        self.vars = {}
        self.output = []
        self.line_count = 0
        self.history = []
    
    def execute(self, code: str) -> dict:
        self.line_count += 1
        try:
            env = {
                '__builtins__': {
                    'int': int, 'str': str, 'float': float, 
                    'bool': bool, 'list': list, 'dict': dict
                },
                'print': self._capture_print
            }
            env.update(self.vars)
            
            exec(code, env, self.vars)
            self.history.append((self.line_count, code))
            
            return {
                'state': self.vars.copy(),
                'output': '\n'.join(self.output),
                'line': self.line_count
            }
        except Exception as e:
            return {'error': str(e), 'line': self.line_count}

    def _capture_print(self, *args):
        self.output.append(' '.join(str(arg) for arg in args))