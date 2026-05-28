from fastmcp import FastMCP

mcp= FastMCP()

@mcp.tool()
def calculate(exp: str)-> str:
    '''calculate the expression and return the result'''

    return str(eval(exp))


@mcp.tool()
def reader(path :str)-> str:
    '''read a file and return its contents'''

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


if __name__ == "__main__":
    mcp.run(transport = "streamable-http", host = "0.0.0.0", port = 8050)