from pydantic import BaseModel, Field
from StructuredAgent import StructuredAgent

# Define the structured output of the model
class DynamoIndex(BaseModel):
    INDEX_NAME: str = Field(description="Name of the index")
    DESCRIPTION: str = Field(description="Description of the index")
    KEYS: str = Field(description="Keys for the index")

class DynamoAttribute(BaseModel):
    ATTRIBUTE_NAME: str = Field(description="Name of the attribute")
    DESCRIPTION: str = Field(description="Description of the attribute")
    TYPE: str = Field(description="Type of the attribute")

class DynamoTable(BaseModel):
    TABLE_NAME: str = Field(description="Name of the table")
    DESCRIPTION: str = Field(description="Description of the table")
    PRIMARY_KEY: str = Field(description="Primary key for the table")
    SORT_KEY: str = Field(description="Sort key for the table")
    INDEXES: list[DynamoIndex] = Field(description="Indexes for the table")
    ATTRIBUTES: list[DynamoAttribute] = Field(description="Attributes for the table")

class DynamoTables(BaseModel):
    TABLES: list[DynamoTable] = Field(description="Tables for the database")

class DynamoDBArchitectAgent:
    def __init__(self, model):
        system_message_template = """
You are an expert at designing databases in DynamoDB for AWS.  
You will be given the description of an API Gateway and it's endpoints.  
You will then design the schema for the database that will support that API.
Your design will include the tables, their primary keys, sort keys and indexes that are needed to support the API requirements.

Description of the API:

{description}

API Endpoints:

{endpoints}
"""
        self._agent = StructuredAgent(model, system_message_template, DynamoTables)

    def create_design(self, system_description, endpoints) -> DynamoTables:
        return self._agent.reply("Create the Database Design", {"description": system_description, "endpoints": endpoints})