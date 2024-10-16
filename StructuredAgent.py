from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

class StructuredAgent:
    def __init__(self, model: BaseChatModel, system_message_template: str, return_type: type):
        self._model = model
        self._system_message_template = system_message_template
        self._return_type = return_type

    def reply(self, prompt: str, merge_data: dict):
        system_message = SystemMessage(content=self._system_message_template.format(**merge_data))
        human_message = HumanMessage(content=prompt.format(**merge_data))
        llm_with_structure = self._model.with_structured_output(self._return_type)
        response = llm_with_structure.invoke([system_message]+[human_message])

        return response
