from fastapi import FastAPI


def create_app():
    app = FastAPI()

    @app.get("/")
    def root():
        return {"Hello": "World"}

    return app


app = create_app()
