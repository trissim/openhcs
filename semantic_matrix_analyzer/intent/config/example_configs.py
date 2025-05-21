"""
Example configurations for the Structural Intent Analysis system.
"""

# Default configuration
DEFAULT_CONFIG = {
    "name_analysis": {
        "enabled": True,
        "tokenization": {
            "separators": ["_", "-", " "],
            "normalize_tokens": True
        },
        "semantic_extraction": {
            "action_verbs": {
                "get": "Retrieve or access",
                "set": "Modify or update",
                "create": "Create or instantiate",
                "delete": "Remove or destroy",
                "update": "Modify or change",
                "validate": "Check or verify",
                "process": "Handle or transform",
                "calculate": "Compute or determine",
                "find": "Search or locate",
                "check": "Verify or test",
                "is": "Test condition",
                "has": "Test possession",
                "can": "Test capability",
                "should": "Test recommendation",
                "will": "Indicate future action",
                "do": "Perform action"
            },
            "design_patterns": {
                "factory": "Create objects",
                "builder": "Construct complex objects",
                "singleton": "Ensure single instance",
                "adapter": "Convert interface",
                "decorator": "Add responsibilities",
                "observer": "Notify of changes",
                "strategy": "Define algorithm family",
                "command": "Encapsulate request",
                "iterator": "Access elements",
                "composite": "Treat objects uniformly",
                "proxy": "Control access",
                "facade": "Simplify interface"
            },
            "domain_objects": {
                "user": "User or account",
                "customer": "Client or buyer",
                "order": "Purchase or request",
                "product": "Item or good",
                "service": "Functionality or offering",
                "transaction": "Exchange or operation",
                "payment": "Financial transaction",
                "account": "User profile or financial account",
                "message": "Communication or notification",
                "event": "Occurrence or happening",
                "request": "Ask or demand",
                "response": "Answer or reply",
                "data": "Information or content",
                "config": "Configuration or settings",
                "manager": "Controller or supervisor",
                "handler": "Processor or responder",
                "provider": "Supplier or source",
                "consumer": "User or recipient"
            }
        },
        "confidence": {
            "base_confidence": 0.5,
            "compound_name_bonus": 0.1,
            "meaningful_token_bonus": 0.2,
            "class_name_bonus": 0.1,
            "method_name_bonus": 0.1
        }
    },
    "type_analysis": {
        "enabled": True,
        "type_mappings": {
            "str": ["String", "Textual data", "entity"],
            "int": ["Integer", "Numeric data", "entity"],
            "float": ["Float", "Numeric data with decimal precision", "entity"],
            "bool": ["Boolean", "True/False condition", "state"],
            "list": ["List", "Collection of items", "entity"],
            "dict": ["Dictionary", "Key-value mapping", "entity"],
            "set": ["Set", "Unique collection of items", "entity"],
            "tuple": ["Tuple", "Immutable collection of items", "entity"],
            "None": ["None", "No value", "state"],
            "Any": ["Any", "Any type", "entity"],
            "Optional": ["Optional", "May be None", "state"],
            "Union": ["Union", "One of several types", "entity"],
            "Callable": ["Callable", "Function or method", "action"],
            "Iterator": ["Iterator", "Sequence that can be iterated", "entity"],
            "Iterable": ["Iterable", "Can be iterated over", "entity"],
            "Generator": ["Generator", "Generates values on demand", "action"],
            "Type": ["Type", "Class or type object", "entity"],
            "Path": ["Path", "File system path", "entity"],
            "datetime": ["Datetime", "Date and time", "entity"],
            "date": ["Date", "Calendar date", "entity"],
            "time": ["Time", "Time of day", "entity"],
            "timedelta": ["Timedelta", "Duration", "entity"],
            "Exception": ["Exception", "Error condition", "state"],
            "Pattern": ["Pattern", "Regular expression pattern", "entity"],
            "Match": ["Match", "Regular expression match", "entity"]
        },
        "confidence": {
            "base_confidence": 0.6,
            "union_optional_bonus": 0.1,
            "collection_bonus": 0.1,
            "custom_type_bonus": 0.2
        }
    },
    "structural_analysis": {
        "enabled": True,
        "patterns": {
            "layered_architecture": {
                "enabled": True,
                "layer_names": ["presentation", "ui", "application", "service", "domain", "model", "data", "persistence", "infrastructure"],
                "confidence": 0.7
            },
            "microservices_architecture": {
                "enabled": True,
                "service_indicators": ["service", "api", "client", "server"],
                "confidence": 0.6
            },
            "event_driven_architecture": {
                "enabled": True,
                "event_indicators": ["event", "listener", "handler", "subscriber", "publisher"],
                "confidence": 0.6
            },
            "mvc_architecture": {
                "enabled": True,
                "model_indicators": ["model", "entity", "domain"],
                "view_indicators": ["view", "template", "page", "screen"],
                "controller_indicators": ["controller", "handler"],
                "confidence": 0.7
            },
            "repository_pattern": {
                "enabled": True,
                "repository_indicators": ["repository", "repo", "dao", "data_access"],
                "confidence": 0.6
            },
            "factory_pattern": {
                "enabled": True,
                "factory_indicators": ["factory", "creator", "builder"],
                "confidence": 0.6
            },
            "singleton_pattern": {
                "enabled": True,
                "singleton_indicators": ["singleton", "instance"],
                "confidence": 0.6
            }
        }
    },
    "integration": {
        "combine_intents": True,
        "build_hierarchy": True,
        "report_format": "text",
        "min_confidence": 0.3,
        "max_results": 100
    }
}

# Python codebase configuration
PYTHON_CONFIG = DEFAULT_CONFIG.copy()
PYTHON_CONFIG["name_analysis"]["tokenization"]["separators"] = ["_", "-", " "]  # Python uses snake_case
PYTHON_CONFIG["type_analysis"]["type_mappings"].update({
    "List": ["List", "Collection of items", "entity"],
    "Dict": ["Dictionary", "Key-value mapping", "entity"],
    "Set": ["Set", "Unique collection of items", "entity"],
    "Tuple": ["Tuple", "Immutable collection of items", "entity"],
    "Optional": ["Optional", "May be None", "state"],
    "Union": ["Union", "One of several types", "entity"],
    "Callable": ["Callable", "Function or method", "action"],
    "Iterator": ["Iterator", "Sequence that can be iterated", "entity"],
    "Iterable": ["Iterable", "Can be iterated over", "entity"],
    "Generator": ["Generator", "Generates values on demand", "action"],
    "Type": ["Type", "Class or type object", "entity"],
    "Path": ["Path", "File system path", "entity"],
    "datetime": ["Datetime", "Date and time", "entity"],
    "date": ["Date", "Calendar date", "entity"],
    "time": ["Time", "Time of day", "entity"],
    "timedelta": ["Timedelta", "Duration", "entity"],
    "Exception": ["Exception", "Error condition", "state"],
    "Pattern": ["Pattern", "Regular expression pattern", "entity"],
    "Match": ["Match", "Regular expression match", "entity"]
})

# Java codebase configuration
JAVA_CONFIG = DEFAULT_CONFIG.copy()
JAVA_CONFIG["name_analysis"]["tokenization"]["separators"] = []  # Java uses camelCase
JAVA_CONFIG["type_analysis"]["type_mappings"] = {
    "String": ["String", "Textual data", "entity"],
    "Integer": ["Integer", "Numeric data", "entity"],
    "int": ["Integer", "Numeric data", "entity"],
    "Long": ["Long", "Numeric data", "entity"],
    "long": ["Long", "Numeric data", "entity"],
    "Double": ["Double", "Numeric data with decimal precision", "entity"],
    "double": ["Double", "Numeric data with decimal precision", "entity"],
    "Float": ["Float", "Numeric data with decimal precision", "entity"],
    "float": ["Float", "Numeric data with decimal precision", "entity"],
    "Boolean": ["Boolean", "True/False condition", "state"],
    "boolean": ["Boolean", "True/False condition", "state"],
    "List": ["List", "Collection of items", "entity"],
    "ArrayList": ["ArrayList", "Collection of items", "entity"],
    "Map": ["Map", "Key-value mapping", "entity"],
    "HashMap": ["HashMap", "Key-value mapping", "entity"],
    "Set": ["Set", "Unique collection of items", "entity"],
    "HashSet": ["HashSet", "Unique collection of items", "entity"],
    "Optional": ["Optional", "May be None", "state"],
    "Callable": ["Callable", "Function or method", "action"],
    "Iterator": ["Iterator", "Sequence that can be iterated", "entity"],
    "Iterable": ["Iterable", "Can be iterated over", "entity"],
    "Stream": ["Stream", "Sequence of elements", "entity"],
    "Class": ["Class", "Class or type object", "entity"],
    "Path": ["Path", "File system path", "entity"],
    "Date": ["Date", "Calendar date", "entity"],
    "LocalDate": ["LocalDate", "Calendar date", "entity"],
    "LocalTime": ["LocalTime", "Time of day", "entity"],
    "LocalDateTime": ["LocalDateTime", "Date and time", "entity"],
    "Duration": ["Duration", "Duration", "entity"],
    "Exception": ["Exception", "Error condition", "state"],
    "Pattern": ["Pattern", "Regular expression pattern", "entity"],
    "Matcher": ["Matcher", "Regular expression match", "entity"]
}

# Microservices architecture configuration
MICROSERVICES_CONFIG = DEFAULT_CONFIG.copy()
MICROSERVICES_CONFIG["structural_analysis"]["patterns"]["microservices_architecture"]["enabled"] = True
MICROSERVICES_CONFIG["structural_analysis"]["patterns"]["microservices_architecture"]["confidence"] = 0.8
MICROSERVICES_CONFIG["structural_analysis"]["patterns"]["microservices_architecture"]["service_indicators"] = [
    "service", "api", "client", "server", "gateway", "proxy", "router", "registry", "discovery", "config", "auth"
]

# Minimal analysis configuration
MINIMAL_CONFIG = {
    "name_analysis": {
        "enabled": True,
        "tokenization": {
            "separators": ["_", "-", " "]
        },
        "semantic_extraction": {
            "action_verbs": {
                "get": "Retrieve or access",
                "set": "Modify or update",
                "create": "Create or instantiate",
                "delete": "Remove or destroy",
                "update": "Modify or change"
            },
            "design_patterns": {},
            "domain_objects": {}
        },
        "confidence": {
            "base_confidence": 0.5
        }
    },
    "type_analysis": {
        "enabled": False
    },
    "structural_analysis": {
        "enabled": False
    },
    "integration": {
        "combine_intents": True,
        "build_hierarchy": False,
        "report_format": "text",
        "min_confidence": 0.5,
        "max_results": 50
    }
}

# Comprehensive analysis configuration
COMPREHENSIVE_CONFIG = DEFAULT_CONFIG.copy()
COMPREHENSIVE_CONFIG["name_analysis"]["semantic_extraction"]["action_verbs"].update({
    "add": "Add or append",
    "remove": "Remove or delete",
    "clear": "Clear or reset",
    "initialize": "Initialize or set up",
    "configure": "Configure or set up",
    "start": "Start or begin",
    "stop": "Stop or end",
    "pause": "Pause or suspend",
    "resume": "Resume or continue",
    "load": "Load or read",
    "save": "Save or write",
    "parse": "Parse or analyze",
    "format": "Format or structure",
    "convert": "Convert or transform",
    "validate": "Validate or check",
    "verify": "Verify or confirm",
    "authenticate": "Authenticate or verify identity",
    "authorize": "Authorize or grant permission",
    "encrypt": "Encrypt or secure",
    "decrypt": "Decrypt or unsecure",
    "compress": "Compress or reduce size",
    "decompress": "Decompress or expand",
    "serialize": "Serialize or convert to data format",
    "deserialize": "Deserialize or convert from data format",
    "encode": "Encode or convert to format",
    "decode": "Decode or convert from format",
    "map": "Map or transform",
    "filter": "Filter or select",
    "reduce": "Reduce or aggregate",
    "sort": "Sort or order",
    "merge": "Merge or combine",
    "split": "Split or divide",
    "join": "Join or combine",
    "connect": "Connect or link",
    "disconnect": "Disconnect or unlink",
    "register": "Register or record",
    "unregister": "Unregister or remove record",
    "subscribe": "Subscribe or listen",
    "unsubscribe": "Unsubscribe or stop listening",
    "publish": "Publish or broadcast",
    "notify": "Notify or inform",
    "handle": "Handle or process",
    "dispatch": "Dispatch or send",
    "receive": "Receive or accept",
    "send": "Send or transmit",
    "request": "Request or ask",
    "respond": "Respond or reply",
    "query": "Query or search",
    "fetch": "Fetch or retrieve",
    "store": "Store or save",
    "cache": "Cache or store temporarily",
    "flush": "Flush or clear cache",
    "backup": "Backup or save copy",
    "restore": "Restore or recover",
    "import": "Import or bring in",
    "export": "Export or send out",
    "compile": "Compile or build",
    "execute": "Execute or run",
    "deploy": "Deploy or install",
    "undeploy": "Undeploy or uninstall",
    "migrate": "Migrate or move",
    "upgrade": "Upgrade or improve",
    "downgrade": "Downgrade or revert",
    "scale": "Scale or resize",
    "monitor": "Monitor or watch",
    "log": "Log or record",
    "trace": "Trace or track",
    "debug": "Debug or troubleshoot",
    "test": "Test or verify",
    "benchmark": "Benchmark or measure performance",
    "profile": "Profile or analyze performance",
    "optimize": "Optimize or improve performance",
    "clean": "Clean or tidy up",
    "shutdown": "Shutdown or close"
})
COMPREHENSIVE_CONFIG["integration"]["min_confidence"] = 0.2
COMPREHENSIVE_CONFIG["integration"]["max_results"] = 500
