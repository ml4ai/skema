from fastapi import FastAPI, Response, status
from fastapi.responses import PlainTextResponse
import os
from skema.rest import (
    schema,
    workflows,
    integrated_text_reading_proxy,
    morae_proxy,
    metal_proxy,
    llm_proxy,
)
from skema.img2mml import eqn2mml
from skema.skema_py import server as code2fn
from skema.program_analysis.comment_extractor import server as comment_service

VERSION: str = os.environ.get("APP_VERSION", "????")

# markdown
description = """
REST API for the SKEMA system
"""

contact = {"name": "SKEMA team", "url": "https://github.com/ml4ai/skema/issues"}

tags_metadata = [
    {
        "name": "core",
        "description": "Status checks, versions, etc.",
        "externalDocs": {
            "description": "Issues",
            "url": "https://github.com/ml4ai/skema/issues?q=is%3Aopen+is%3Aissue+label%3AIntegration",
        },
    },
    {
        "name": "workflows",
        "description": "Operations related to end-to-end SKEMA workflows",
        "externalDocs": {
            "description": "Issues",
            "url": "https://github.com/ml4ai/skema/issues?q=is%3Aopen+is%3Aissue+label%3AIntegration",
        },
    },
    {
        "name": "code2fn",
        "description": "Operations to transform source code into function networks.",
        "externalDocs": {
            "description": "Issues",
            "url": "https://github.com/ml4ai/skema/issues?q=is%3Aopen+is%3Aissue+label%3ACode2FN",
        },
    },
    {
        "name": "eqn2mml",
        "description": "Operations to transform equations",
        "externalDocs": {
            "description": "Issues",
            "url": "https://github.com/ml4ai/skema/issues?q=is%3Aopen+is%3Aissue+label%3AEquations",
        },
    },
    {
        "name": "morae",
        "description": "",
        "externalDocs": {
            "description": "Issues",
            "url": "https://github.com/ml4ai/skema/issues?q=is%3Aopen+is%3Aissue+label%3AMORAE",
        },
    },
    {
        "name": "text reading",
        "description": "Unified proxy and integration code for MIT and SKEMA TR pipelines",
    },
    {
        "name": "metal",
        "description": "AMR linking endpoints",
    },
]

app = FastAPI(
    version=VERSION,
    title="SKEMA API",
    description=description,
    contact=contact,
    openapi_tags=tags_metadata,
)

app.include_router(
    workflows.router,
    prefix="/workflows",
    tags=["workflows"],
)

app.include_router(
    eqn2mml.router,
    prefix="/eqn2mml",
    tags=["eqn2mml"],
)

app.include_router(
    code2fn.router,
    prefix="/code2fn",
    tags=["code2fn"],
)

app.include_router(
    comment_service.router,
    prefix="/code2fn",
    tags=["code2fn"],
)

app.include_router(
    morae_proxy.router,
    prefix="/morae",
    tags=["morae", "skema-rs"],
)

app.include_router(
    llm_proxy.router,
    prefix="/morae",
    tags=["morae"],
)

app.include_router(
    integrated_text_reading_proxy.router,
    prefix="/text-reading",
    tags=["text reading"]
)

app.include_router(
    metal_proxy.router,
    prefix="/metal",
    tags=["metal"]
)


@app.get("/version", tags=["core"], summary="API version")
async def version() -> str:
    return PlainTextResponse(VERSION)


@app.get(
    "/healthcheck",
    tags=["core"],
    summary="Health of component services",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "model": schema.HealthStatus,
            "description": "All component services are healthy (200 status)",
        },
        500: {
            "model": schema.HealthStatus,
            "description": "One or more component services are unhealthy (non-200 status)",
        },
    },
)
async def healthcheck(response: Response) -> schema.HealthStatus:
    morae_status = await morae_proxy.healthcheck()
    mathjax_status = eqn2mml.latex2mml_healthcheck()
    eqn2mml_status = eqn2mml.img2mml_healthcheck()
    code2fn_status = code2fn.ping()
    text_reading_status = integrated_text_reading_proxy.healthcheck()
    metal_status = metal_proxy.healthcheck()
    # check if any services failing and alter response status code accordingly
    status_code = (
        status.HTTP_200_OK
        if all(
            code == 200
            for code in [
                morae_status,
                mathjax_status,
                eqn2mml_status,
                code2fn_status,
                text_reading_status,
                metal_status
            ]
        )
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    response.status_code = status_code
    return schema.HealthStatus(
        morae=morae_status,
        mathjax=mathjax_status,
        eqn2mml=eqn2mml_status,
        code2fn=code2fn_status,
        text_reading=text_reading_status,
        metal=metal_status
    )
