from CodeBaseModels import CodeFile, CodeReview
from StructuredAgent import StructuredAgent

class DynamoDBTerraformAgent:
    def __init__(self, model):
        writer_system_message_template = """
You are an expert at writting Terraform.  You will be given a database design.  
Review the design and generate a corresponding Terraform script that will create a DynamoDB table on AWS. 
Ensure the table's attributes, keys, and indexes are correctly represented based on the design.
You will omit any "provider" or "terraform" blocks from the script, they will be supplied by a different script.

Given the following information, you will create or improve a Terraform script that will create a DynamoDB table on AWS:

Database Design:

{design}

Current Script: 

{code}

Code Review: 

{review}
"""

        reviewer_system_message_template = """
You are an expert at reviewing Terraform scripts that are used to create DynamoDB tables on AWS.  
You will review the Terraform script that has been written to create a DynamoDB table and provide feedback on how it can be improved or corrected.  
You will provide a review of the script and a score from 1 to 10, with 10 being the best.

Given the following information, you will review the Terraform script and give feedback on how it can be improved or corrected.

Database Design:

{design}

Current Terraform Script:

{script}
"""

        self._writer_agent = StructuredAgent(model, writer_system_message_template, CodeFile)
        self._reviewer_agent = StructuredAgent(model, reviewer_system_message_template, CodeReview)

    def write_terraform(self, design, min_quality_score: int = 8, max_review_iterations: int = 3) -> CodeFile:
        review = ""
        code = ""
        review_score = 0
        review_count = 0

        while (review_score < min_quality_score and review_count < max_review_iterations):
            terraform_script: CodeFile = self._writer_agent.reply("Create or improve the Terraform Script", {"design": design, "code": code, "review": review})

            code = terraform_script.RAW_CODE
            
            if terraform_script:
                terraform_review: CodeReview = self._reviewer_agent.reply("Review the Terraform Script", {"design": design, "script": code})
                review = terraform_review.REVIEW
                review_score = terraform_review.SCORE
                review_count += 1

        return terraform_script