from httpx import AsyncClient
from skema.rest.workflows import app
import pytest
import json


@pytest.mark.ci_only
@pytest.mark.asyncio
async def test_post_eqns_to_mets_mathml_latex():
    """
    Test case for /equations-to-met.
    """
    latex_equations = [
        """
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <mi>E</mi>
          <mo>=</mo>
          <mi>m</mi>
          <msup>
            <mi>c</mi>
            <mn>2</mn>
          </msup>
        </math>
        """,
        "c=\\frac{a}{b}",
    ]

    endpoint = "/equations-to-met"

    async with AsyncClient(app=app, base_url="http://latex-to-mets-test") as ac:
        response = await ac.post(endpoint, json={"equations": latex_equations})
    expected = [
        {
            "Cons": [
                "Equals",
                [
                    {
                        "Atom": {
                            "Ci": {
                                "type": None,
                                "content": {"Mi": "E"},
                                "func_of": None,
                                "notation": None,
                            }
                        }
                    },
                    {
                        "Cons": [
                            "Multiply",
                            [
                                {
                                    "Atom": {
                                        "Ci": {
                                            "type": None,
                                            "content": {"Mi": "m"},
                                            "func_of": None,
                                            "notation": None,
                                        }
                                    }
                                },
                                {
                                    "Cons": [
                                        "Power",
                                        [
                                            {
                                                "Atom": {
                                                    "Ci": {
                                                        "type": None,
                                                        "content": {"Mi": "c"},
                                                        "func_of": None,
                                                        "notation": None,
                                                    }
                                                }
                                            },
                                            {"Atom": {"Mn": "2"}},
                                        ],
                                    ]
                                },
                            ],
                        ]
                    },
                ],
            ]
        },
        {
            "Cons": [
                "Equals",
                [
                    {
                        "Atom": {
                            "Ci": {
                                "type": None,
                                "content": {"Mi": "c"},
                                "func_of": None,
                                "notation": None,
                            }
                        }
                    },
                    {
                        "Cons": [
                            "Divide",
                            [
                                {
                                    "Atom": {
                                        "Ci": {
                                            "type": None,
                                            "content": {"Mi": "a"},
                                            "func_of": None,
                                            "notation": None,
                                        }
                                    }
                                },
                                {
                                    "Atom": {
                                        "Ci": {
                                            "type": None,
                                            "content": {"Mi": "b"},
                                            "func_of": None,
                                            "notation": None,
                                        }
                                    }
                                },
                            ],
                        ]
                    },
                ],
            ]
        },
    ]

    # check for route's existence
    assert (
        any(route.path == endpoint for route in app.routes) == True
    ), "{endpoint} does not exist for app"
    # check status code
    assert (
        response.status_code == 200
    ), f"Request was unsuccessful (status code was {response.status_code} instead of 200)"
    # check response
    assert (
        json.loads(response.text) == expected
    ), f"Response should be {expected}, but instead received {response.text}"


@pytest.mark.ci_only
@pytest.mark.asyncio
async def test_post_eqns_to_mets_latex_mathml():
    """
    Test case for /equations-to-met.
    """
    mathml_equations = [
        "E=mc^2",
        """
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <mi>c</mi>
          <mo>=</mo>
          <mfrac>
            <mrow>
              <mi>a</mi>
            </mrow>
            <mrow>
              <mi>b</mi>
            </mrow>
          </mfrac>
        </math>
        """,
    ]

    endpoint = "/equations-to-met"

    async with AsyncClient(app=app, base_url="http://mathml-to-mets-test") as ac:
        response = await ac.post(endpoint, json={"equations": mathml_equations})
    expected = [
        {
            "Cons": [
                "Equals",
                [
                    {
                        "Atom": {
                            "Ci": {
                                "type": None,
                                "content": {"Mi": "E"},
                                "func_of": None,
                                "notation": None,
                            }
                        }
                    },
                    {
                        "Cons": [
                            "Multiply",
                            [
                                {
                                    "Atom": {
                                        "Ci": {
                                            "type": None,
                                            "content": {"Mi": "m"},
                                            "func_of": None,
                                            "notation": None,
                                        }
                                    }
                                },
                                {
                                    "Cons": [
                                        "Power",
                                        [
                                            {
                                                "Atom": {
                                                    "Ci": {
                                                        "type": None,
                                                        "content": {"Mi": "c"},
                                                        "func_of": None,
                                                        "notation": None,
                                                    }
                                                }
                                            },
                                            {"Atom": {"Mn": "2"}},
                                        ],
                                    ]
                                },
                            ],
                        ]
                    },
                ],
            ]
        },
        {
            "Cons": [
                "Equals",
                [
                    {
                        "Atom": {
                            "Ci": {
                                "type": None,
                                "content": {"Mi": "c"},
                                "func_of": None,
                                "notation": None,
                            }
                        }
                    },
                    {
                        "Cons": [
                            "Divide",
                            [
                                {
                                    "Atom": {
                                        "Ci": {
                                            "type": None,
                                            "content": {"Mi": "a"},
                                            "func_of": None,
                                            "notation": None,
                                        }
                                    }
                                },
                                {
                                    "Atom": {
                                        "Ci": {
                                            "type": None,
                                            "content": {"Mi": "b"},
                                            "func_of": None,
                                            "notation": None,
                                        }
                                    }
                                },
                            ],
                        ]
                    },
                ],
            ]
        },
    ]

    # check for route's existence
    assert (
        any(route.path == endpoint for route in app.routes) == True
    ), "{endpoint} does not exist for app"
    # check status code
    assert (
        response.status_code == 200
    ), f"Request was unsuccessful (status code was {response.status_code} instead of 200)"
    # check response
    assert (
        json.loads(response.text) == expected
    ), f"Response should be {expected}, but instead received {response.text}"