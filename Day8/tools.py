from langchain_core.tools import tool

@tool
def calculator(expression:str) -> str:
    """
    Performs mathematical calculations
    """

    try:
        result = eval(expression)
        return result

    except Exception as e:
        return f"Error: {str(e)}"