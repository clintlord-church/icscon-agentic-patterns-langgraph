from CodeBaseModels import CodeFile, CodeReview
from StructuredAgent import StructuredAgent

class APIGatewayTerraformAgent:
    def __init__(self, model):
        writer_system_message_template = """
You are an expert at writing Terraform.  You will be given an API definition that includes the endpoints, request and response parameters, and data storage requirements.  
Review the API definition and generate a corresponding Terraform script that will create an API Gateway on AWS.  
Ensure the endpoints, request and response parameters, and data storage requirements are correctly represented based on the API definition.

Follow these rules:

1. There will be existing lambda function for each endpoint, you do not need to write terraform for the lambda functions
2. Connect each endpoint to the corresponding lambda function, The lambda function's name will be the same as the endpoint's name
3. Each endpoint will use the same authorizer lambda called "endpoint_authorizer", which already exists
4. Do not add a provider block to the terraform script, it will be supplied by another existing script
5. Use the OpenAPI Specification to define the API Gateway

Given the following information, you will create or improve a Terraform script that will create an API Gateway on AWS:

API Endpoints:

{endpoints}

Current Script:

{code}

Code Review:

{review}
"""

        reviewer_system_message_template = """
You are an expert at reviewing Terraform scripts that are used to create API Gateways on AWS.
You will review the Terraform script that has been written to create an API Gateway and provide feedback on how it can be improved or corrected.
You will provide a review of the script and a score from 1 to 10, with 10 being the best.

The Terraform should follow these rules:

1. There will be existing lambda function for each endpoint, you do not need to write terraform for the lambda functions
2. Connect each endpoint to the corresponding lambda function, The lambda function's name will be the same as the endpoint's name
3. Each endpoint will use the same authorizer lambda called "endpoint_authorizer", which already exists
4. Do not add a provider block to the terraform script, it will be supplied by another existing script
5. Use the OpenAPI Specification to define the API Gateway

Given the following information, you will review the Terraform script and give feedback on how it can be improved or corrected.

API Endpoints:

{endpoints}

Current Terraform Script:

{script}
"""

        self._writer_agent = StructuredAgent(model, writer_system_message_template, CodeFile)
        self._reviewer_agent = StructuredAgent(model, reviewer_system_message_template, CodeReview)

    def write_terraform(self, endpoints, min_quality_score: int = 8, max_review_iterations: int = 3) -> CodeFile:
        review = ""
        code = ""
        review_score = 0
        review_count = 0

        while (review_score < min_quality_score and review_count < max_review_iterations):
            terraform_script: CodeFile = self._writer_agent.reply("Create or improve the Terraform Script", {"endpoints": endpoints, "code": code, "review": review})

            code = terraform_script.RAW_CODE
            
            if terraform_script:
                terraform_review: CodeReview = self._reviewer_agent.reply("Review the Terraform Script", {"endpoints": endpoints, "script": code})
                review = terraform_review.REVIEW
                review_score = terraform_review.SCORE
                review_count += 1
        return terraform_script