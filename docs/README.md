Source code for the SKEMA website.

Requires the Haskell Tool Stack, which can be installed using
[GHCup](https://www.haskell.org/ghcup/).

Once you have Stack installed, run the following commands in this directory.

```
stack build
```

This will build the `generator` executable, which uses the
[Hakyll](https://jaspervdj.be/hakyll/) library for static site generation.
Then, execute the following command:

```
make build
```

This will build the website. The following command will start a local
development server that live-updates the website as you make changes to it
(requires refreshing the page in the browser).

```
make watch
```
