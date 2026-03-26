REQUERY_PROMPT = """You previously solved a math problem and got a different answer than other models for step {step_number}.

The problem was:
{problem_text}

Your solution for step {step_number} was:
{your_step}

However, {agree_count} out of {total_count} other models agree that the result for this step should be:
{consensus_result}

Please carefully re-examine your work for this step. If you made an error, correct it. If you believe your answer is correct, explain why.

Respond using the exact same format:
STEP {step_number}: [description]
MATH: [expression]
RESULT: [result]"""


DEBATE_PROMPT = """Multiple AI models solved the same math problem and got different results for step {step_number}.

The problem was:
{problem_text}

Here are the different solutions for this step:

{all_solutions}

Analyze which solution is mathematically correct and explain why. Then provide the correct solution.

Respond with:
CORRECT MODEL: [model name or "none"]
EXPLANATION: [why that solution is correct]
STEP {step_number}: [description]
MATH: [expression]
RESULT: [correct result]"""
