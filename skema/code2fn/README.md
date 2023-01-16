Code2FN
=======

The Code2FN web service ingests source code and returns a GroMEt function
network module collection.

There are two ways to run the service. The first is to use the Dockerized
version (see `../../README.md`). To run the service 'unDockerized', you can
invoke the following command:

```
uvicorn server:app --reload
```

The command above assumes you have installed the `skema` package as detailed in
`../../README.md` .

We also provide a client script to test the service (`client.py`). Run the
following to see the command line options for `client.py`:

```
./client.py -h
```
