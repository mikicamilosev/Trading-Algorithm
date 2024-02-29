from fastapi import FastAPI

# Create an instance of FastAPI
app = FastAPI()

# Define a route for the root endpoint
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

# Define a route with path parameter
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}