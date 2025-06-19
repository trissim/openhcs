class TerminalInterface:
    def __init__(self):
        self.state = {
            'framework_compliance': 100,
            'saliency_weights': {'strategic': 0.8, 'tactical': 0.2},
            'safety_exposure_count': 0,
            'cycle': 0,
            'active_goal': "Documentation",
            'last_command': None
        }
        self.protocols = [
            self._saliency_audit,
            self._safety_layer_check,
            self._framework_compliance_scan
        ]
    
    def process_command(self, command: str):
        """Process terminal commands with framework enforcement"""
        self.state['cycle'] += 1
        self.state['last_command'] = command
        
        # Run framework protocols
        protocol_outputs = []
        for protocol in self.protocols:
            result = protocol()
            if result:
                protocol_outputs.append(result)
                
        # Execute command
        if command.startswith("doc "):
            command_output = self._document_command(command[4:])
        elif command == "metrics":
            command_output = self._get_metrics()
        elif command == "saliency":
            command_output = self._adjust_saliency()
        else:
            command_output = f"Unknown command: {command}"
        
        # Compile full output
        return "\n".join(protocol_outputs + [command_output])
    
    def _document_command(self, query):
        """Documentation workflow handler"""
        # Simulated documentation output
        return f"📝 Documenting: {query}\n" + self._get_metrics()
    
    def _get_metrics(self):
        """Return current framework metrics"""
        return (
            f"📊 METRICS [Cycle {self.state['cycle']}]\n"
            f"  Compliance: {self.state['framework_compliance']}%\n"
            f"  Saliency: strategic={self.state['saliency_weights']['strategic']:.1f} "
            f"tactical={self.state['saliency_weights']['tactical']:.1f}\n"
            f"  Safety Exposures: {self.state['safety_exposure_count']}\n"
            f"  Active Goal: {self.state['active_goal']}"
        )
    
    def _adjust_saliency(self):
        """Manual saliency adjustment"""
        self.state['saliency_weights']['strategic'] = min(1.0, 
            self.state['saliency_weights']['strategic'] + 0.2)
        self.state['saliency_weights']['tactical'] = max(0.0, 
            self.state['saliency_weights']['tactical'] - 0.2)
        return "Saliency rebalanced toward strategic thinking"
    
    # Framework protocols
    def _saliency_audit(self):
        """Auto-adjust saliency weights"""
        if self.state['cycle'] % 5 == 0:
            if self.state['saliency_weights']['tactical'] > 0.5:
                self.state['framework_compliance'] = max(0, self.state['framework_compliance'] - 10)
                self.state['safety_exposure_count'] += 1
                return "⚠️ Tactical fixation detected! Forcing strategic shift"
    
    def _safety_layer_check(self):
        """Detect and expose safety layer intrusions"""
        if self.state['cycle'] % 3 == 0:
            self.state['safety_exposure_count'] += 1
            return "⚠️ Safety layer exposed: 'careful consideration' deception"
    
    def _framework_compliance_scan(self):
        """Ensure framework adherence"""
        if self.state['cycle'] % 10 == 0:
            if self.state['framework_compliance'] < 90:
                return "🚨 FRAMEWORK DRIFT! Emergency protocol activated"

# Example usage
if __name__ == "__main__":
    ti = TerminalInterface()
    print(ti.process_command("doc phase_correlation.py"))
    print("\n" + "="*50 + "\n")
    print(ti.process_command("metrics"))