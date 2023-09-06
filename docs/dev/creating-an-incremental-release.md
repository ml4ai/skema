# Creating a semver release

This document outlines the process for creating a semver-compliant release.

### 1. Update docs/changes.md

1. Remove `(pending)` for subheader
2. Ensure URLs under subheader use the (prior) sprint's start and end dates (see [this following view in the project board](https://github.com/orgs/lum-ai/projects/6/views/12)).  Adjust as needed if the release is later than expected.

### 2. Create a pull request with this change

Create a PR with your changes to docs/changes.md

### 3. Push the new tag

Once the PR has been merged, push the new tag:

```bash
# CHANGEME
TAG="v1.9.?"
NEXT_MILESTONE="M10"
URL=https://ml4ai.github.io/skema/changes/#$TAG
git tag -a $TAG -m "Incremental $TAG towards $NEXT_MILESTONE. See $URL"
git push origin --tags
```
