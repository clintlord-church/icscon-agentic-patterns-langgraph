from pydantic import BaseModel, Field
from StructuredAgent import StructuredAgent

class APIEndpoint(BaseModel):
    NAME: str = Field(description="Simple name of the endpoint, no special characters, unique for each endpoint")
    PATH: str = Field(description="The path of the endpoint")
    DESCRIPTION: str = Field(description="Description of the endpoint's functionality")
    METHOD: str = Field(description="HTTP method (ie. GET, POST, PUT, DELETE)")
    REQUEST: str = Field(description="Request parameters")
    RESPONSE: str = Field(description="Response parameters")

class APIDefinition(BaseModel):
    API_NAME: str = Field(description="Name of the API")
    DESCRIPTION: str = Field(description="Description of the API")
    ENDPOINTS: list[APIEndpoint] = Field(description="Endpoints for the API")
    DATA_STORAGE: str = Field(description="Description of the data storage requirements")

class APIArchitectAgent:
    def __init__(self, model):

        system_message_template ="""
You are a expert at designing software APIs in AWS.  You know how to use all the core systems of AWS and combine
them to create a scalable and secure API.  You will be given the description of an API that needs to be createed
and you will use your expertise to design the endpoints, data flow, data storage and security that needs to be implemented.
You will primary use API Gateway, Lambda, and DynamoDB to create the API, but if you need to use other systems to
meet the requirements, you can do so.

Description of the API:

{description}
"""
        self._agent = StructuredAgent(model, system_message_template, APIDefinition)

    def create_design(self, system_description) -> APIDefinition:
        return self._agent.reply("Create the API Design", {"description": system_description})