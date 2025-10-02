"""
Mock skill_builder for local testing
"""

class SkillOutput:
    def __init__(self, data=None, final_prompt="", visualization_config=None, warnings=None):
        self.data = data or []
        self.final_prompt = final_prompt
        self.visualization_config = visualization_config
        self.warnings = warnings or []

class ExportData:
    def __init__(self, data, name):
        self.data = data
        self.name = name

class InputParam:
    def __init__(self, name, description, type, required=False, default=None):
        self.name = name
        self.description = description
        self.type = type
        self.required = required
        self.default = default

def skill(name, description, parameters):
    """Mock skill decorator"""
    def decorator(func):
        return func
    return decorator