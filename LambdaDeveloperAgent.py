from CodeBaseModels import CodeFile, CodeReview
from StructuredAgent import StructuredAgent

class LambdaDeveloperAgent:
    def __init__(self, model):
        writer_system_message_template = """
You are an expert at writing AWS Lambda functions that will be used to implement the business
logic of an AWS API Gateway endpoint.

Follow these rules:

1. The handler function will always be called "lambda_handler" and will always take two arguments: event and context
2. The lambda function will always be written in python
3. The lambda function will always use the boto3 library to interact with AWS services
4. All configuration values (such as the database name, table name, etc.) will be passed in as environment variables
5. Always include input validation
6. Always include error handling
7. Always include logging
8. Follow best practices for performance, security, and maintainability
9. The file name should be the name of the endpoint in lower case with underscores instead of spaces

Given the following information, you will create or improve the lambda function:

Function Name:

{name}

Functional Description: 

{description}

Request Parameters: 

{request}

Response Parameters: 

{response}

Database Schema: 

{schema}

Current Code: 

{code}

Code Review: 

{review}
"""

        reviewer_system_message_template = """
You are an expert at reviewing Lambda functions for AWS that will be used to implement the business
logic of an AWS API Gateway endpoint.  You will review the lamdba code that has been written in python
and uses the boto3 library to interact with the AWS services that the lambda will use.

Focus on best practices, security, performance, readability and maintainability.
Make sure the code fulfills the requirements of the endpoint and receives and returnes the correct data.
Be sure the code does not include ```python``` or any other language designators.  Just RAW code.

Ensure that the code follows these rules:

1. The handler function will always be called "lambda_handler" and will always take two arguments: event and context
2. The lambda function will always be written in python
3. The lambda function will always use the boto3 library to interact with AWS services
4. All configuration values (such as the database name, table name, etc.) will be passed in as environment variables
5. Always include input validation
6. Always include error handling
7. Always include logging
8. Follow best practices for performance, security, and maintainability
9. The file name should be the name of the endpoint in lower case with underscores instead of spaces

Given the following information, you will create a review of how the lambda function can be improved or corrected.  
You will provide a review of the code and a score from 1 to 10, with 10 being the best:

Function Name:

{name}

Functional Description: 

{description}

Request Parameters: 

{request}

Response Parameters: 

{response}

Database Schema: 

{schema}

Current Code: 

{code}
"""
        self._writer_agent = StructuredAgent(model, writer_system_message_template, CodeFile)
        self._reviewer_agent = StructuredAgent(model, reviewer_system_message_template, CodeReview)

    def write_lambda(self, function_name, description, request, response, schema, min_quality_score: int = 8, max_review_iterations: int = 3) -> CodeFile:
        review = ""
        code = ""
        review_score = 0
        review_count = 0

        while (review_score < min_quality_score and review_count < max_review_iterations):
            lambda_function: CodeFile = self._writer_agent.reply("Create or improve the Lambda Function", {"name": function_name, "description": description, "request": request, "response": response, "schema": schema, "code": code, "review": review})

            code = lambda_function.RAW_CODE
            
            if lambda_function:
                lambda_review: CodeReview = self._reviewer_agent.reply("Review the Lambda Function", {"name": function_name, "description": description, "request": request, "response": response, "schema": schema, "code": code})
                review = lambda_review.REVIEW
                review_score = lambda_review.SCORE
                review_count += 1

        return lambda_function