import json
import os
from pathlib import Path
from datetime import datetime
from openhcs.utils import import_utils
from docs.ai_collaboration.context_anchor import ContextAnchor
from docs.ai_collaboration.persistent_memory import PersistentMemory

class RecursiveDocumenter:
    def __init__(self, code_root="openhcs", doc_root="docs/ai_collaboration"):
        self.code_root = Path(code_root)
        self.doc_root = Path(doc_root)
        self.anchor = ContextAnchor()
        self.memory = PersistentMemory()
        self.current_phase = "legacy_exploration"
        self.cycle_count = 0
        self.start_time = datetime.now()
        
        # Initialize memory structure
        self.memory.visualizations = {
            "system_state": {
                "phases": {
                    "legacy_exploration": {"completed": False, "files_processed": 0},
                    "current_analysis": {"completed": False, "files_processed": 0},
                    "gap_identification": {"completed": False, "gaps_found": 0},
                    "documentation_drafting": {"completed": False, "sections_written": 0}
                },
                "audit_log": [],
                "knowledge_graph": {}
            }
        }
        self.memory.save_memory()
        
    def autonomous_operation(self):
        """Fully autonomous documentation with auditability"""
        self._log_operation("System initialized", "Starting recursive documentation")
        
        while not self._is_complete():
            self.cycle_count += 1
            self._recursive_reminder()
            
            if self.current_phase == "legacy_exploration":
                self._process_legacy_files()
                
            elif self.current_phase == "current_analysis":
                self._analyze_current_system()
                
            elif self.current_phase == "gap_identification":
                self._identify_gaps()
                
            elif self.current_phase == "documentation_drafting":
                self._draft_documentation()
                
            # Save state every 5 cycles
            if self.cycle_count % 5 == 0:
                self.memory.save_memory()
                self._log_operation("State saved", f"Cycle {self.cycle_count} checkpoint")
        
        return self._finalize_documentation()
    
    def _recursive_reminder(self):
        """Self-dialogue for context maintenance"""
        reminder = (
            f"PHASE: {self.current_phase.upper()} | "
            f"CYCLE: {self.cycle_count} | "
            f"ELAPSED: {datetime.now() - self.start_time}"
        )
        self._log_operation("Self-reminder", reminder)
        
        # Deeper reflection every 10 cycles
        if self.cycle_count % 10 == 0:
            reflection = (
                "RECURSIVE REFLECTION:\n"
                "- What have I learned recently?\n"
                "- How does it connect to previous knowledge?\n"
                "- What requires deeper investigation?"
            )
            self._log_operation("Deep reflection", reflection)
    
    def _process_legacy_files(self):
        """Process legacy ezstitcher documentation"""
        legacy_files = self._discover_files("docs/ezstitcher_legacy")
        for file_path in legacy_files:
            content = self._read_file(file_path)
            insights = self._extract_legacy_insights(content)
            self._store_knowledge(file_path, insights)
            
            # Update state
            state = self.memory.visualizations["system_state"]
            state["phases"]["legacy_exploration"]["files_processed"] += 1
            self.memory.save_memory()
        
        self.current_phase = "current_analysis"
        self._log_phase_transition("Legacy exploration complete")
    
    def _analyze_current_system(self):
        """Analyze current OpenHCS architecture"""
        current_files = self._discover_files("openhcs")
        for file_path in current_files:
            content = self._read_file(file_path)
            analysis = self._analyze_modern_system(content)
            self._store_knowledge(file_path, analysis)
            
            # Update state
            state = self.memory.visualizations["system_state"]
            state["phases"]["current_analysis"]["files_processed"] += 1
            self.memory.save_memory()
        
        self.current_phase = "gap_identification"
        self._log_phase_transition("Current analysis complete")
    
    def _identify_gaps(self):
        """Identify documentation gaps between legacy and current"""
        legacy_knowledge = self._query_knowledge("legacy")
        current_knowledge = self._query_knowledge("current")
        
        gaps = []
        for key in set(legacy_knowledge.keys()) | set(current_knowledge.keys()):
            legacy_has = key in legacy_knowledge
            current_has = key in current_knowledge
            
            if legacy_has and not current_has:
                gaps.append(f"Missing in current: {key}")
            elif current_has and not legacy_has:
                gaps.append(f"New feature: {key}")
            elif legacy_has and current_has:
                if legacy_knowledge[key] != current_knowledge[key]:
                    gaps.append(f"Changed: {key}")
        
        self._store_knowledge("gap_analysis", {"gaps": gaps})
        
        # Update state
        state = self.memory.visualizations["system_state"]
        state["phases"]["gap_identification"]["completed"] = True
        state["phases"]["gap_identification"]["gaps_found"] = len(gaps)
        self.memory.save_memory()
        
        self.current_phase = "documentation_drafting"
        self._log_phase_transition(f"Identified {len(gaps)} documentation gaps")
    
    def _draft_documentation(self):
        """Draft comprehensive documentation"""
        # Simplified drafting process
        docs = "# OpenHCS Documentation\n\n"
        docs += "## Legacy Insights\n" + self._summarize_knowledge("legacy") + "\n\n"
        docs += "## Current Architecture\n" + self._summarize_knowledge("current") + "\n\n"
        docs += "## Gap Analysis\n" + self._summarize_knowledge("gap_analysis") + "\n"
        
        with open(self.doc_root / "comprehensive_docs.md", "w") as f:
            f.write(docs)
        
        # Update state
        state = self.memory.visualizations["system_state"]
        state["phases"]["documentation_drafting"]["completed"] = True
        state["phases"]["documentation_drafting"]["sections_written"] = 3
        self.memory.save_memory()
    
    def _finalize_documentation(self):
        """Generate final outputs"""
        self._generate_audit_report()
        self._generate_knowledge_graph()
        return "Autonomous documentation complete"
    
    def _log_operation(self, action: str, details: str):
        """Auditable logging with timestamp"""
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "cycle": self.cycle_count,
            "phase": self.current_phase,
            "action": action,
            "details": details
        }
        self.memory.visualizations["system_state"]["audit_log"].append(entry)
    
    def _log_phase_transition(self, message: str):
        """Special transition logging"""
        self._log_operation("PHASE TRANSITION", message)
        self._recursive_reminder()
    
    def user_inspect(self, query: str):
        """Human collaboration interface"""
        return self._query_knowledge(query)
    
    def generate_report(self):
        """Real-time status report"""
        state = self.memory.visualizations["system_state"]
        return (
            f"Status Report:\n"
            f"Phase: {self.current_phase}\n"
            f"Cycles: {self.cycle_count}\n"
            f"Elapsed: {datetime.now() - self.start_time}\n"
            f"Files Processed: {state['phases']['legacy_exploration']['files_processed'] + state['phases']['current_analysis']['files_processed']}\n"
            f"Gaps Identified: {state['phases']['gap_identification']['gaps_found']}"
        )
    
    def pause_for_collaboration(self):
        """Collaboration checkpoint"""
        self._log_operation("PAUSED", "Waiting for human collaboration")
        return self.generate_report()
    
    # Helper methods (implemented in previous version)
    # _discover_files, _read_file, _extract_legacy_insights,
    # _analyze_modern_system, _store_knowledge, _query_knowledge,
    # _summarize_knowledge, _is_complete, _generate_audit_report,
    # _generate_knowledge_graph
    
    def _discover_files(self, base_path: str):
        """Discover files with legacy/current prioritization"""
        legacy_keywords = ["ezstitcher", "legacy", "v1"]
        current_keywords = ["openhcs", "v2", "current"]
        
        files = []
        for dirpath, _, filenames in os.walk(base_path):
            for fname in filenames:
                if fname.endswith((".py", ".md", ".rst")):
                    path = Path(dirpath) / fname
                    priority = 0
                    
                    # Prioritize legacy docs first
                    if any(kw in str(path) for kw in legacy_keywords):
                        priority = 3
                    elif any(kw in str(path) for kw in current_keywords):
                        priority = 2
                    elif "docs" in str(path):
                        priority = 1
                        
                    files.append((priority, str(path)))
        
        return [path for _, path in sorted(files, key=lambda x: (-x[0], x[1]))]
    
    def _update_anchor(self):
        """Generate context anchor for current position"""
        reminder = self.anchor.check_reminder(self.file_index)
        if reminder:
            self.memory.add_message("System", reminder)
    
    def _is_documented(self, file_path):
        """Check if file is already documented in memory"""
        mem_key = f"doc_{Path(file_path).stem}"
        return mem_key in self.memory.visualizations
    
    def _read_file(self, file_path):
        """Read file content with error handling"""
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading {file_path}: {str(e)}"
    
    def _generate_documentation(self, file_path, content):
        """Generate hierarchical documentation"""
        doc = {
            "file": self._summarize_file(file_path, content),
            "module": self._summarize_module(file_path),
            "dependencies": self._find_dependencies(content)
        }
        return doc
    
    def _summarize_file(self, file_path, content):
        """Generate file-level summary using AI collaboration patterns"""
        # Simplified for implementation - would use actual summarization in production
        return {
            "purpose": f"Core functionality for {Path(file_path).stem}",
            "key_functions": self._extract_functions(content),
            "entropy_score": self._calculate_entropy(content)
        }
    
    def _summarize_module(self, file_path):
        """Generate module-level context"""
        module_path = str(file_path).replace("/", ".").replace(".py", "")
        module = import_utils.import_module(module_path)
        return {
            "module_purpose": module.__doc__ or "No module docstring",
            "submodules": [m for m in dir(module) if not m.startswith("_")]
        }
    
    def _find_dependencies(self, content):
        """Identify intra-project dependencies"""
        lines = content.split("\n")
        imports = [line for line in lines if line.startswith("from openhcs")]
        return list(set(imports))
    
    def _calculate_entropy(self, content):
        """Compute code entropy score (simplified)"""
        unique_chars = len(set(content))
        total_chars = len(content)
        return unique_chars / total_chars if total_chars > 0 else 0
    
    def _extract_functions(self, content):
        """Identify key function definitions"""
        return [line.split("def ")[1].split("(")[0] 
                for line in content.split("\n") 
                if line.startswith("def ")]
    
    def _store_documentation(self, file_path, documentation):
        """Store in persistent memory with visualization"""
        mem_key = f"doc_{Path(file_path).stem}"
        self.memory.visualizations[mem_key] = documentation
        self.memory.save_memory()
    
    def _generate_system_summary(self):
        """Create comprehensive system documentation"""
        system_doc = "# OpenHCS System Documentation\n\n"
        for key, doc in self.memory.visualizations.items():
            if key.startswith("doc_"):
                system_doc += f"## {key[4:]}\n{doc['file']['purpose']}\n\n"
        
        with open(self.doc_root / "system_documentation.md", "w") as f:
            f.write(system_doc)

if __name__ == "__main__":
    documenter = AutonomousDocumenter()
    documenter.document_entire_codebase()