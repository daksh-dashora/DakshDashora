from fastmcp import FastMCP

mcp= FastMCP()

@mcp.tool()
def fetch():
    '''fetch the data'''

    return{
        'data': 'hello mcp!'
    }


if __name__ == "__main__":
    mcp.run(transport = "stdio")


    