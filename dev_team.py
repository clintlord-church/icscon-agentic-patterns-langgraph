import os
import datetime
from typing import Annotated
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.types import Send
from IPython.display import Image
from APIArchitectAgent import APIDefinition, APIArchitectAgent
from DynamoDBArchitectAgent import DynamoTables, DynamoDBArchitectAgent
from CodeBaseModels import CodeFile
from LambdaDeveloperAgent import LambdaDeveloperAgent
from DynamoDBTerraformAgent import DynamoDBTerraformAgent
from APIGatewayTerraformAgent import APIGatewayTerraformAgent

# model used for planning and other general cognative tasks
general_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# model used for coding
coding_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# setting for code review
min_quality_score = 6
max_review_iterations = 1


def add_codefile(left: list[CodeFile], right: list[CodeFile]) -> list[CodeFile]:
    for r in right:
        left.append(r)

    return left


# Define the state details
class DevTeamState(MessagesState):
    SystemDescription: str
    APIDefinition: APIDefinition
    APIGatewayTerraformScript: CodeFile
    DatabaseArchitecture: DynamoTables
    DatabaseTerraformScript: CodeFile
    CurrentEndpointIndex: int
    LambdaFunctionList: Annotated[list[CodeFile], add_codefile]


# Define the function that calls the model
def architect_api(state: DevTeamState):
    # extract data from the state
    system_description = state['SystemDescription']

    # call the agent
    api_architect = APIArchitectAgent(general_model)
    api_definition = api_definition = api_architect.create_design(system_description)

    # update the state
    return {"APIDefinition": api_definition}


def design_database(state: DevTeamState):
    # extract data from the state
    system_description = state['SystemDescription']
    endpoints = state['APIDefinition'].ENDPOINTS
    endpoint_list = [e.model_dump_json() for e in endpoints]

    # call the agent
    dynamodb_architect = DynamoDBArchitectAgent(general_model)
    database_architecture = dynamodb_architect.create_design(system_description, endpoint_list)

    # update the state
    return {"DatabaseArchitecture": database_architecture}


def write_database_terraform(state: DevTeamState):
    # extract data from the state
    database_design: DynamoTables = state['DatabaseArchitecture']
    database_table_list = [dd.model_dump_json() for dd in database_design.TABLES]

    # call the agent
    dynamo_terraform_writer = DynamoDBTerraformAgent(coding_model)
    terraform_script = dynamo_terraform_writer.write_terraform(database_table_list, min_quality_score, max_review_iterations)

    # update the state
    return {"DatabaseTerraformScript": terraform_script}


def write_apigateway_terraform(state: DevTeamState):
    # extract data from the state
    endpoint_list = [e.model_dump_json() for e in state['APIDefinition'].ENDPOINTS]

    # call the agent
    api_gateway_terraform_writer = APIGatewayTerraformAgent(coding_model)
    terraform_script = api_gateway_terraform_writer.write_terraform(endpoint_list, min_quality_score, max_review_iterations)

    # update the state
    return {"APIGatewayTerraformScript": terraform_script}


def develop_lambda(state: DevTeamState):
    # extract data from the state
    endpoint = state['APIDefinition'].ENDPOINTS[state['CurrentEndpointIndex']]
    database_design = state['DatabaseArchitecture']
    database_table_list = [dd.model_dump_json() for dd in database_design.TABLES]

    # call the agent
    lambda_developer = LambdaDeveloperAgent(coding_model)
    lambda_function = lambda_developer.write_lambda(endpoint.NAME, endpoint.DESCRIPTION, endpoint.REQUEST, endpoint.RESPONSE, database_table_list, min_quality_score, max_review_iterations)

    # update the state
    return {"LambdaFunctionList": [lambda_function]}

    
def send_to_developer(state: DevTeamState):
    result = []
    for index in range(len(state['APIDefinition'].ENDPOINTS)):
        state_copy = state.copy()
        state_copy['CurrentEndpointIndex'] = index
        result.append(Send("lambda_developer_agent", state_copy))

    return result


# Define a new graph
workflow = StateGraph(DevTeamState)

# Define the two nodes we will cycle between
workflow.add_node("api_architect_agent", architect_api)
workflow.add_node("database_architect_agent", design_database)
workflow.add_node("lambda_developer_agent", develop_lambda)
workflow.add_node("database_terraform_writer_agent", write_database_terraform)
workflow.add_node("api_gateway_terraform_writer_agent", write_apigateway_terraform)

workflow.add_edge(START, "api_architect_agent")
workflow.add_edge("api_architect_agent", "database_architect_agent")
workflow.add_edge("api_architect_agent", "api_gateway_terraform_writer_agent")
workflow.add_edge("database_architect_agent", "database_terraform_writer_agent")
workflow.add_edge("api_gateway_terraform_writer_agent", END)
workflow.add_edge("database_terraform_writer_agent", END)
workflow.add_conditional_edges("database_architect_agent", send_to_developer, ["lambda_developer_agent"])
workflow.add_edge("lambda_developer_agent", END)

app = workflow.compile()

# get the current running folder
running_folder = os.path.dirname(os.path.abspath(__file__))

# set the current folder to the dev folder + todays date in YYYYMMDDhhmmss format
dev_folder = running_folder + "/dev/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# create the folder if it does not exist
if not os.path.exists(dev_folder):
    os.makedirs(dev_folder, exist_ok=True)

# save the png of the graph
image = Image(app.get_graph(xray=1).draw_mermaid_png())
open(f"{dev_folder}/dev_team_graph.png", "wb").write(image.data)

description = """
Build an API that will allow the user to create, read, update and delete blog posts.
Each blog post should have a title, content, author, date created and average rating.
Besides the basic CRUD operations, the API should also allow the user to:

1. Submit a rating for a blog post 
2. Retrieve the average rating for a post
3. Search for blog posts by author or by date
"""

# for s in app.stream( 
#     {"messages": [HumanMessage(content=description)], "SystemDescription": description},
#     config={"configurable": {"thread_id": 42}, "recursion_limit": 1000}):
#     print(s)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content=description)], "SystemDescription": description},
    config={"configurable": {"thread_id": 42}, "recursion_limit": 1000}
)

try:
    # save the API definition to a file
    with open(f"{dev_folder}/api_definition.json", "w") as f:
        api_definition: APIDefinition = final_state['APIDefinition']
        f.write(api_definition.model_dump_json(indent=4))
except:
    pass                

try:
    # save the database schema to a file
    with open(f"{dev_folder}/database_schema.json", "w") as f:
        database_architecture: DynamoTables = final_state['DatabaseArchitecture']
        f.write(database_architecture.model_dump_json(indent=4))
except:
    pass

try:
    # save the terraform script to a file
    terraform_script: CodeFile = final_state['APIGatewayTerraformScript']
    with open(f"{dev_folder}/APIGateway.tf", "w") as f:
        f.write(terraform_script.RAW_CODE)
except:
    pass

try:
    # save the terraform script to a file
    terraform_script: CodeFile = final_state['DatabaseTerraformScript']
    with open(f"{dev_folder}/Database.tf", "w") as f:
        f.write(terraform_script.RAW_CODE)
except:
    pass

try:
    # save the individual lambda functions to files
    lambda_functions: list[CodeFile] = final_state['LambdaFunctionList']
    for lambda_function in lambda_functions:
        with open(f"{dev_folder}/{lambda_function.FILENAME}", "w") as f:
            f.write(lambda_function.RAW_CODE)
except:
    pass